import gc
import logging
import time
import threading
from pathlib import Path
import platform
from typing import Optional, TYPE_CHECKING

import sounddevice as sd

from .audio_recorder import AudioRecorder
from .whisper_engine import WhisperEngine, create_whisper_engine
from .runtime_options import (FASTER_WHISPER, INSTALLABLE, ONNX_ASR, UNSUPPORTED, WHISPER_CPP, Runtime, best_onnx_runtime,
                              choose_compute_type, current_runtime_key, detect_runtimes, runtimes_for_engine_family,
                              with_whisper_cpp_paths)
from .runtime_loader import CPU, ONNX_CUDA, ONNX_DIRECTML, VULKAN, whisper_cpp_runtime_paths
from .model_registry import ONNX_FAMILY
from .platform import dialogs
from .clipboard_manager import ClipboardManager
from .system_tray import SystemTray
from .config_manager import ConfigManager
from .text_postprocessor import TextPostProcessor
from .audio_feedback import AudioFeedback
from .output_audio_control import OutputAudioControl
from .utils import OptionalComponent
from .voice_activity_detection import VadEvent, VadManager
from .voice_commands import VoiceCommandManager
from .terminal_title import TerminalTitle

if TYPE_CHECKING:
    from .floating_widget import FloatingWidget

class StateManager:
    MIN_VAD_HYSTERESIS_GAP = 0.05
    MIN_VAD_OFFSET_THRESHOLD = 0.05
    MAX_VAD_ONSET_THRESHOLD = 0.95
    AUTO_TRIGGER_COOLDOWN_SECONDS = 1.0  # Lets feedback sounds die out before the VAD may start a recording

    def __init__(self,
                 audio_recorder: AudioRecorder,
                 whisper_engine: WhisperEngine,
                 clipboard_manager: ClipboardManager,
                 config_manager: ConfigManager,
                 vad_manager: VadManager,
                 text_postprocessor: TextPostProcessor,
                 system_tray: Optional[SystemTray] = None,
                 audio_feedback: Optional[AudioFeedback] = None,
                 voice_command_manager: Optional[VoiceCommandManager] = None,
                 terminal_title: Optional[TerminalTitle] = None,
                 output_audio_control: Optional[OutputAudioControl] = None):

        self.audio_recorder = audio_recorder
        self.whisper_engine = whisper_engine
        self.clipboard_manager = clipboard_manager
        self.system_tray = OptionalComponent(system_tray)
        self.config_manager = config_manager
        self.audio_feedback = OptionalComponent(audio_feedback)
        self.output_audio_control = OptionalComponent(output_audio_control)
        self.vad_manager = vad_manager
        self.text_postprocessor = text_postprocessor
        self.voice_command_manager = voice_command_manager
        self.terminal_title = OptionalComponent(terminal_title)
        self.floating_widget = OptionalComponent(None)
        self.floating_widget_available = False

        self.is_processing = False
        self.is_model_loading = False
        self.is_muted = False
        self.last_transcription = None
        self._pending_model_change = None
        self._pending_device_change = None
        self._command_mode = False
        self._state_lock = threading.Lock()
        self._recording_stop_lock = threading.Lock()
        self._monitoring_lock = threading.Lock()
        self._toggle_lock = threading.Lock()
        self._mute_lock = threading.Lock()
        self._streaming_display_active = False
        self.auto_trigger_enabled = config_manager.get_setting('vad', 'auto_trigger_enabled')
        self._auto_triggered_recording = False
        self.vad_sensitivity_window = OptionalComponent(None)
        self.vad_sensitivity_window_attached = False
        self._vad_sensitivity_window_open = False
        self._vad_hysteresis_gap = None
        self._auto_trigger_resume_time = 0.0
        self._auto_trigger_paste = config_manager.get_setting('vad', 'auto_trigger_paste')
        self._auto_trigger_lock = threading.Lock()
        self._start_lock = threading.Lock()

        self._runtimes = None
        self._runtime_install_running = False
        self._runtime_installer = None
        self._install_progress_window = None
        self._runtime_key = None
        self._runtimes_lock = threading.Lock()
        self._fallback_runtime = None
        self._failed_runtime_key = None

        self.logger = logging.getLogger(__name__)
        self._current_audio_host = None
        self._initialize_audio_host()

    def attach_components(self,
                          audio_recorder: AudioRecorder,
                          system_tray: Optional[SystemTray],
                          floating_widget: Optional["FloatingWidget"] = None):
        self.audio_recorder = audio_recorder
        self.system_tray = OptionalComponent(system_tray)
        self.floating_widget = OptionalComponent(floating_widget)
        self.floating_widget_available = floating_widget is not None
        self._ensure_audio_device_for_host(self._current_audio_host)
        self._apply_auto_trigger()

    def _update_ui_state(self, state: str):
        if self.is_engine_unavailable() and state != "recording":
            self.system_tray.update_state("processing")
            self.terminal_title.update_state("processing")
            self.floating_widget.update_state("loading")
            return
        if state == "idle" and self.is_muted:
            state = "muted"
        self.system_tray.update_state(state)
        self.terminal_title.update_state(state)
        self.floating_widget.update_state(state)

    def is_engine_unavailable(self) -> bool:
        return self.is_model_loading or self._runtime_install_running

    def _begin_engine_switch(self) -> bool:
        with self._state_lock:
            if self._runtime_install_running:
                return False
            self._runtime_install_running = True
        if self.audio_recorder.get_recording_status():
            print("🎤 Cancelling recording, transcription is paused until the switch completes...")
            self.cancel_active_recording()
        self._update_ui_state("idle")
        return True

    def _end_engine_switch(self, refresh_ui: bool = True):
        with self._state_lock:
            self._runtime_install_running = False
            self._runtime_installer = None
        self.system_tray.set_status_text(None)
        if refresh_ui:
            self._update_ui_state(self.get_current_state())

    def _show_outcome(self, outcome: str):
        self.floating_widget.show_outcome(outcome)

    def handle_max_recording_duration_reached(self, audio_data):
        self.logger.info("Max recording duration reached - starting transcription")
        self._transcription_pipeline(audio_data, use_auto_enter=False)

    def handle_vad_event(self, event: VadEvent):
        if event == VadEvent.SILENCE_TIMEOUT:
            with self._recording_stop_lock:
                if not self.audio_recorder.get_recording_status():
                    return
                self.logger.info("VAD silence timeout detected - stopping recording")
                timeout_seconds = int(self.vad_manager.vad_silence_timeout_seconds)
                self._clear_streaming_display()
                print(f"⏰ Stopping recording after {timeout_seconds} seconds of silence...")
                audio_data = self.audio_recorder.stop_recording()
            self._transcription_pipeline(audio_data, use_auto_enter=False)
        elif event == VadEvent.SPEECH_START:
            self._handle_auto_trigger_speech_start()
        elif event == VadEvent.SPEECH_END:
            self._handle_auto_trigger_speech_end()

    def _handle_auto_trigger_speech_start(self):
        # Don't let our own feedback sounds trigger a new recording
        if time.monotonic() < self._auto_trigger_resume_time:
            self.logger.debug("Auto-trigger: ignoring speech during feedback cooldown")
            return
        with self._start_lock:
            if not self.auto_trigger_enabled or self._vad_sensitivity_window_open or not self.can_start_recording():
                return
            self.logger.info("Auto-trigger: speech detected, starting recording")
            self._begin_recording(auto_triggered=True)

    def _resume_auto_trigger_if_speaking(self):
        detector = self.audio_recorder.continuous_vad
        if detector and detector.is_speech_active():
            self._handle_auto_trigger_speech_start()

    def _start_auto_trigger_cooldown(self):
        self._auto_trigger_resume_time = time.monotonic() + self.AUTO_TRIGGER_COOLDOWN_SECONDS

    def play_ready_sound(self):
        self._start_auto_trigger_cooldown()
        self.audio_feedback.play_ready_sound()

    def _handle_auto_trigger_speech_end(self):
        if not self._auto_triggered_recording or not self.is_transcription_recording():
            return
        self.logger.info("Auto-trigger: speech ended, stopping recording")
        self.stop_recording()

    def attach_vad_sensitivity_window(self, vad_sensitivity_window):
        self.vad_sensitivity_window = OptionalComponent(vad_sensitivity_window)
        self.vad_sensitivity_window_attached = vad_sensitivity_window is not None

    def is_vad_sensitivity_window_available(self) -> bool:
        return self.vad_sensitivity_window_attached and self.audio_recorder.continuous_vad is not None

    def open_vad_sensitivity_window(self):
        self._vad_sensitivity_window_open = True
        self.vad_sensitivity_window.open()

    def handle_vad_sensitivity_window_opened(self):
        self._vad_sensitivity_window_open = True
        self._apply_auto_trigger()

    def handle_vad_sensitivity_window_closed(self):
        self._vad_sensitivity_window_open = False
        threading.Thread(target=self._apply_auto_trigger, daemon=True).start()

    def handle_vad_probability(self, probability: float):
        if self._vad_sensitivity_window_open:
            self.vad_sensitivity_window.update_probability(probability)

    def update_vad_onset_threshold(self, onset_threshold: float):
        onset_threshold = round(min(onset_threshold, self.MAX_VAD_ONSET_THRESHOLD), 2)
        if self._vad_hysteresis_gap is None:
            self._vad_hysteresis_gap = max(self.MIN_VAD_HYSTERESIS_GAP, self.vad_manager.vad_onset_threshold - self.vad_manager.vad_offset_threshold)
        offset_threshold = round(max(self.MIN_VAD_OFFSET_THRESHOLD, onset_threshold - self._vad_hysteresis_gap), 2)
        self.vad_manager.vad_onset_threshold = onset_threshold
        self.vad_manager.vad_offset_threshold = offset_threshold
        if self.audio_recorder.continuous_vad:
            self.audio_recorder.continuous_vad.set_thresholds(onset_threshold, offset_threshold)
        self.config_manager.update_user_setting('vad', 'vad_onset_threshold', onset_threshold)
        self.config_manager.update_user_setting('vad', 'vad_offset_threshold', offset_threshold)

    def _apply_auto_trigger(self):
        with self._monitoring_lock:
            if not (self.auto_trigger_enabled or self._vad_sensitivity_window_open):
                self.audio_recorder.stop_monitoring()
            elif not self.audio_recorder.start_monitoring():
                self.logger.warning("Microphone monitoring needs real-time VAD (vad.vad_realtime_enabled) and ten-vad")
                print("⚠️ Voice-activated recording cannot run: it needs vad.vad_realtime_enabled and ten-vad")
                return False
            return True

    def handle_monitoring_failed(self, error: Exception):
        # Runtime-only switch-off: the saved setting stays on so a restart retries
        self.logger.error(f"Voice-activated recording stopped: {error}")
        print("❌ Voice-activated recording turned off (microphone stream failed). Re-enable it from the tray menu.")
        self.auto_trigger_enabled = False
        self.vad_sensitivity_window.update_probability(0.0)
        self.system_tray.refresh_menu()

    def is_auto_trigger_available(self) -> bool:
        return self.audio_recorder.continuous_vad is not None

    def update_auto_trigger(self, enabled: bool):
        with self._auto_trigger_lock:
            self.auto_trigger_enabled = enabled
            if self._apply_auto_trigger():
                self.config_manager.update_user_setting('vad', 'auto_trigger_enabled', enabled)
                print("🎙️ Voice-activated recording on: the microphone is listening" if enabled else "🎙️ Voice-activated recording off")
            else:
                self.auto_trigger_enabled = False
        self.system_tray.refresh_menu()

    def handle_streaming_result(self, text: str, is_final: bool):
        if is_final:
            if self._streaming_display_active:
                print(f"\r   {text:<70}")
                self._streaming_display_active = False
        else:
            display_text = text if len(text) < 67 else "..." + text[-64:]
            print(f"\r   {display_text:<70}", end="", flush=True)
            self._streaming_display_active = True

    def _clear_streaming_display(self):
        if self._streaming_display_active:
            print("\r" + " " * 75 + "\r", end="", flush=True)
            self._streaming_display_active = False
    
    def stop_recording(self, use_auto_enter: bool = False) -> bool:
        with self._recording_stop_lock:
            if not self.audio_recorder.get_recording_status():
                return False
            self._clear_streaming_display()
            audio_data = self.audio_recorder.stop_recording()

        self._transcription_pipeline(audio_data, use_auto_enter)
        return True
    
    def cancel_active_recording(self):
        self._clear_streaming_display()
        self._command_mode = False
        self.audio_recorder.cancel_recording()
        self.output_audio_control.release()
        self._start_auto_trigger_cooldown()
        self.audio_feedback.play_cancel_sound()
        self._update_ui_state("idle")
    
    def cancel_recording_hotkey_pressed(self) -> bool:
        current_state = self.get_current_state()
        
        if current_state == "recording":
            print("🎤 Recording cancelled!")            
            self.cancel_active_recording()
            return True
        else:
            return False
    
    def start_recording(self):
        with self._start_lock:
            if self.can_start_recording():
                self._begin_recording()
                return

        current_state = self.get_current_state()
        if self.is_muted:
            print("🔇 Microphone is muted - unmute to record")
        elif self.is_processing:
            print("⏳ Still processing previous recording...")
        elif self.is_engine_unavailable():
            print("⏳ Transcription is unavailable until the model is ready...")
        else:
            print(f"⏳ Cannot record while {current_state}...")

    def toggle_recording(self):
        if not self._toggle_lock.acquire(blocking=False):
            return
        try:
            if self.audio_recorder.get_recording_status():
                self.stop_recording()
            else:
                self.start_recording()
        except Exception:
            self.logger.exception("Toggle recording failed")
        finally:
            self._toggle_lock.release()

    def start_command_recording(self):
        with self._start_lock:
            if self.can_start_recording():
                self._begin_command_recording()
            elif self.is_muted:
                print("🔇 Microphone is muted - unmute to record")
            elif self.is_engine_unavailable():
                print("⏳ Transcription is unavailable until the model is ready...")

    def _begin_command_recording(self):
        with self._state_lock:
            self._command_mode = True
            self._auto_triggered_recording = False

        self.logger.info("Starting command mode recording")
        success = self.audio_recorder.start_recording()
        if success and self.is_muted:
            self.audio_recorder.cancel_recording()
            with self._state_lock:
                self._command_mode = False
            print("🔇 Microphone is muted - unmute to record")
            return
        if success:
            print("\n🎤 Command mode activated! Speak a command...")
            self.config_manager.print_command_stop_instructions()
            self.audio_feedback.play_start_sound()
            self.output_audio_control.engage()
            self._update_ui_state("recording")

    def _begin_recording(self, auto_triggered: bool = False):
        # Set before starting so a quick SPEECH_END on another thread already sees it
        self._auto_triggered_recording = auto_triggered
        success = self.audio_recorder.start_recording(voice_activated=auto_triggered)

        if not success:
            self._auto_triggered_recording = False
        elif self.is_muted:
            self._auto_triggered_recording = False
            self.audio_recorder.cancel_recording()
            print("🔇 Microphone is muted - unmute to record")
        else:
            print("\n🎤 Recording started! Speak now...")
            if not auto_triggered:
                self.config_manager.print_stop_instructions_based_on_config()
            self.audio_feedback.play_start_sound()
            self.output_audio_control.engage()
            self._update_ui_state("recording")
    
    def _transcription_pipeline(self, audio_data, use_auto_enter: bool = False):
        fallback_to = None
        self.output_audio_control.release()
        try:
            with self._state_lock:
                if self.is_engine_unavailable():
                    print("⏳ Runtime is switching, recording discarded")
                    return
                self.is_processing = True
                command_mode = self._command_mode
                self._command_mode = False
                engine = self.whisper_engine
                auto_triggered = self._auto_triggered_recording

            self.audio_feedback.play_stop_sound()

            if audio_data is None:
                return

            duration = self.audio_recorder.get_audio_duration(audio_data)
            print(f"   ✓ Recorded {duration:.1f} seconds, transcribing...")

            self._update_ui_state("processing")

            try:
                transcribed_text = engine.transcribe_audio(audio_data)
            except Exception:
                fallback = self._fallback_runtime
                if fallback and engine is self.whisper_engine:
                    self._fallback_runtime = None
                    self._failed_runtime_key = self._runtime_key
                    fallback_to = fallback
                raise
            self._fallback_runtime = None
            self._failed_runtime_key = None

            if not transcribed_text:
                return

            transcribed_text = self.text_postprocessor.process(transcribed_text)
            print(f"   ✓ Transcribed: '{transcribed_text}'")
            self._log_transcription(transcribed_text)

            if command_mode:
                command_ok = self._handle_command_transcription(transcribed_text, use_auto_enter)
                self._show_outcome("success" if command_ok else "error")
                return

            dictation_command = None if auto_triggered and not self._auto_trigger_paste else self._match_dictation_command(transcribed_text)
            if dictation_command:
                command_ok = self.voice_command_manager.execute_command(dictation_command, use_auto_enter)
                self._show_outcome("success" if command_ok else "error")
                return

            if auto_triggered and not self._auto_trigger_paste:
                success = self.clipboard_manager.copy_with_notification(transcribed_text)
            else:
                success = self.clipboard_manager.deliver_transcription(
                    transcribed_text, use_auto_enter
                )

            if success:
                self.last_transcription = transcribed_text
                self.audio_feedback.play_transcription_complete_sound()
            self._show_outcome("success" if success else "error")

        except Exception as e:
            self.logger.error(f"Error in processing workflow: {e}")
            print(f"❌ Error processing recording: {e}")
            self._show_outcome("error")
        
        finally:
            with self._state_lock:
                self.is_processing = False
                self._auto_triggered_recording = False
                self._start_auto_trigger_cooldown()
                pending_model = self._pending_model_change
                pending_device = self._pending_device_change
                self._pending_model_change = None
                self._pending_device_change = None
                if pending_model:
                    self.is_model_loading = True

            if pending_device:
                device_id, device_name = pending_device
                self.logger.info(f"Executing pending device change to: {device_name}")
                self._execute_audio_device_change(device_id, device_name)

            if pending_model:
                self.logger.info(f"Executing pending model change to: {pending_model}")
                print(f"🔄 Processing complete, now switching to [{pending_model}] model...")
                self._execute_model_change(pending_model)

            if not (pending_device or pending_model):
                self._update_ui_state("idle")

            if self._apply_auto_trigger() and self.auto_trigger_enabled:
                threading.Timer(self.AUTO_TRIGGER_COOLDOWN_SECONDS, self._resume_auto_trigger_if_speaking).start()

            if fallback_to and not pending_model:
                runtime, compute_type = fallback_to
                print(f"⚠️ New runtime failed during transcription, falling back to [{runtime.label}]")
                self.request_runtime_change(runtime.key, compute_type)

    def _log_transcription(self, text: str):
        log_config = self.config_manager.get_logging_config()
        if log_config.get('log_transcriptions', False):
            self.logger.info(f"Transcribed text: '{text}'")
        else:
            self.logger.info(f"Transcribed {len(text)} chars")

    def _handle_command_transcription(self, text: str, use_auto_enter: bool = False) -> bool:
        if not self.voice_command_manager.enabled:
            self.logger.warning("Voice commands disabled")
            return False

        matched = self.voice_command_manager.match_command(text)
        if matched:
            return self.voice_command_manager.execute_command(matched, use_auto_enter)
        print("   ✗ No matching command found")
        return False

    def _match_dictation_command(self, text: str) -> Optional[dict]:
        if not (self.voice_command_manager and self.voice_command_manager.match_in_dictation):
            return None
        matched = self.voice_command_manager.match_command(text)
        if matched:
            print(f"   ✓ Voice command matched: '{matched.get('trigger', '')}'")
        return matched

    def get_application_state(self) -> dict:
        status = {
            "recording": self.audio_recorder.get_recording_status(),
            "processing": self.is_processing,
            "model_loading": self.is_engine_unavailable(),
        }
        
        return status
    
    def manual_transcribe_test(self, duration_seconds: int = 5):
        try:
            print(f"🎤 Recording for {duration_seconds} seconds...")
            print("Speak now!")
            
            self.audio_recorder.start_recording()
            
            time.sleep(duration_seconds)
            
            audio_data = self.audio_recorder.stop_recording()
            self._transcription_pipeline(audio_data)
            
        except Exception as e:
            self.logger.error(f"Manual test failed: {e}")
            print(f"❌ Test failed: {e}")
    
    def shutdown(self):        
        print("Whisper Key is shutting down... goodbye!")

        if self.audio_recorder.get_recording_status():
            self.audio_recorder.stop_recording()
        self.output_audio_control.release_blocking()
        self.vad_sensitivity_window.stop()
        self.audio_recorder.stop_monitoring()

        installer = self._runtime_installer
        if installer:
            installer.cancel()

        self.system_tray.stop()
        self.terminal_title.stop()
        self.floating_widget.stop()
        try:
            self.whisper_engine.close()
        except Exception as e:
            self.logger.error(f"Failed to close the transcription engine: {e}")
    
    def set_model_loading(self, loading: bool):
        with self._state_lock:
            old_state = self.is_model_loading
            self.is_model_loading = loading

        if old_state != loading:
            self._update_ui_state("processing" if loading else "idle")
    
    def is_transcription_recording(self) -> bool:
        return self.audio_recorder.get_recording_status() and not self._command_mode

    def can_start_recording(self) -> bool:
        with self._state_lock:
            return not (self.is_muted or self.is_processing or self.is_engine_unavailable() or self.audio_recorder.get_recording_status())
    
    def get_current_state(self) -> str:
        with self._state_lock:
            if self.is_engine_unavailable():
                return "model_loading"
            elif self.is_processing:
                return "processing"
            elif self.audio_recorder.get_recording_status():
                return "recording"
            else:
                return "idle"
    
    def request_model_change(self, new_model_key: str) -> bool:
        if new_model_key == self.whisper_engine.model_key:
            return True

        language = self.config_manager.get_setting('whisper', 'language')
        if self.whisper_engine.registry.is_english_only(new_model_key) and language not in (None, 'auto', 'en'):
            self.logger.warning(f"Model {new_model_key} is English-only, the configured language [{language}] is ignored")
            print(f"⚠️ [{new_model_key}] only transcribes English, the configured language [{language}] is ignored")

        download_state = self.get_model_download_state(new_model_key)
        if download_state == "download":
            if not self._begin_engine_switch():
                print("⏳ A download is already running...")
                return False
            threading.Thread(target=self._download_model_and_switch, args=(new_model_key,), daemon=True).start()
            return True
        if download_state == "unavailable":
            print(f"⚠️ The [{new_model_key}] model is not available for the current runtime")
            return False

        if self._is_onnx_model(new_model_key) != self._is_onnx_engine():
            return self._switch_engine_family(new_model_key)

        if self.get_current_state() == "recording":
            print(f"🎤 Cancelling recording to switch to [{new_model_key}] model...")
            self.cancel_active_recording()

        with self._state_lock:
            if self.is_model_loading:
                print("⏳ Model already loading, please wait...")
                return False
            if self.is_processing:
                print(f"⏳ Queueing model change to [{new_model_key}] until transcription completes...")
                self._pending_model_change = new_model_key
                return True
            self.is_model_loading = True

        self._update_ui_state("processing")
        self._execute_model_change(new_model_key)
        return True

    def toggle_mute(self):
        self.set_muted(not self.is_muted)

    def set_muted(self, muted: bool):
        with self._mute_lock:
            with self._state_lock:
                if self.is_muted == muted:
                    return
                self.is_muted = muted
            if muted:
                print("🔇 Microphone muted - recording disabled")
                if self.audio_recorder.get_recording_status():
                    self.cancel_active_recording()
            else:
                print("🎤 Microphone unmuted")
            self.floating_widget.set_muted(muted)
            self._update_ui_state(self.get_current_state())

    def update_floating_widget_enabled(self, enabled: bool):
        self.config_manager.update_user_setting('floating_widget', 'enabled', enabled)
        if enabled:
            self.floating_widget.show()
        else:
            self.floating_widget.hide()

    def preview_floating_widget_size(self, size: float):
        self.floating_widget.set_size(size)

    def update_floating_widget_size(self, size: float):
        self.config_manager.update_user_setting('floating_widget', 'size', size)
        self.floating_widget.set_size(size)

    def update_floating_widget_save_position(self, save_position: bool):
        self.config_manager.update_user_setting('floating_widget', 'save_position', save_position)
        self.floating_widget.set_save_position(save_position)

    def save_floating_widget_position(self, position: str):
        self.config_manager.update_user_setting('floating_widget', 'position', position)

    def save_floating_widget_locked(self, locked: bool):
        self.config_manager.update_user_setting('floating_widget', 'locked', locked)

    def save_floating_widget_size(self, size: float):
        self.config_manager.update_user_setting('floating_widget', 'size', size)
        self.system_tray.refresh_menu()

    def save_orb_position(self, position: str):
        self.config_manager.update_user_setting('floating_widget', 'orb_position', position)

    def update_floating_widget_style(self, style: str):
        self.config_manager.update_user_setting('floating_widget', 'style', style)
        self.floating_widget.set_style(style)

    def update_orb_skin(self, skin: str):
        self.config_manager.update_user_setting('floating_widget', 'orb_skin', skin)
        self.floating_widget.set_orb_skin(skin)

    def update_orb_locked(self, locked: bool):
        self.config_manager.update_user_setting('floating_widget', 'locked', locked)
        self.floating_widget.set_orb_locked(locked)
        self.system_tray.refresh_menu()

    def update_orb_buttons(self, show_lock_button: bool, show_mute_button: bool):
        self.config_manager.update_user_setting('floating_widget', 'orb_lock_button', show_lock_button)
        self.config_manager.update_user_setting('floating_widget', 'orb_mute_button', show_mute_button)
        self.floating_widget.set_orb_buttons(show_lock_button, show_mute_button)

    def update_orb_hide_on_fullscreen(self, hide_on_fullscreen: bool):
        self.config_manager.update_user_setting('floating_widget', 'orb_hide_on_fullscreen', hide_on_fullscreen)
        self.floating_widget.set_orb_hide_on_fullscreen(hide_on_fullscreen)

    def preview_orb_appearance(self, opacity: float, vibrancy: float, hue: float):
        self.floating_widget.set_orb_appearance(opacity, vibrancy, hue)

    def update_orb_appearance(self, opacity: float, vibrancy: float, hue: float):
        self.config_manager.update_user_setting('floating_widget', 'orb_opacity', opacity)
        self.config_manager.update_user_setting('floating_widget', 'orb_vibrancy', vibrancy)
        self.config_manager.update_user_setting('floating_widget', 'orb_hue', hue)
        self.floating_widget.set_orb_appearance(opacity, vibrancy, hue)

    def _ggml_model_dir(self) -> Optional[str]:
        return with_whisper_cpp_paths(self.config_manager.get_whisper_config()).get('cpp_model_dir')

    def _is_onnx_engine(self) -> bool:
        return self.whisper_engine.ENGINE_TYPE == ONNX_ASR

    def _is_onnx_model(self, model_key: str) -> bool:
        return self.whisper_engine.registry.get_engine_family(model_key) == ONNX_FAMILY

    def _whisper_models_use_whisper_cpp(self) -> bool:
        engine_type = self.whisper_engine.ENGINE_TYPE
        if engine_type == ONNX_ASR:
            return self.config_manager.get_setting('whisper', 'runtime') == VULKAN
        return engine_type == WHISPER_CPP

    def get_model_download_state(self, model_key: str) -> Optional[str]:
        registry = self.whisper_engine.registry
        if self._is_onnx_model(model_key):
            return "ready" if registry.is_onnx_model_downloaded(model_key) else "download"
        if not self._whisper_models_use_whisper_cpp():
            if registry.is_model_cached(model_key):
                return "ready"
            return "download" if registry.get_hf_repo(model_key) else "unavailable"
        engine = self.whisper_engine
        if engine.ENGINE_TYPE == WHISPER_CPP and engine._is_model_cached(model_key):
            return "ready"
        from .runtime_installer import ggml_model_name
        model_dir = self._ggml_model_dir()
        ggml_name = ggml_model_name(model_key)
        if model_dir and ggml_name and (Path(model_dir) / f"ggml-{ggml_name}.bin").is_file():
            return "ready"
        if model_dir and ggml_name:
            return "download"
        return "unavailable"

    def _download_model_and_switch(self, model_key: str):
        from .runtime_installer import RuntimeInstaller

        registry = self.whisper_engine.registry
        model = registry.get_model(model_key)
        label = model.label if model else model_key
        title = f"Download {label}"
        message = f"{label} is not downloaded yet. Download it now?\n\nTranscription is paused until the download completes."
        if self._is_onnx_model(model_key):
            download = lambda installer: installer.download_onnx_model(registry, model_key)
        elif self._whisper_models_use_whisper_cpp():
            title = "Download whisper.cpp model"
            message = f"The whisper.cpp file for the [{model_key}] model is not downloaded yet. Download it now?"
            download = lambda installer: installer.download_ggml_model(model_key, Path(self._ggml_model_dir()))
        else:
            download = lambda installer: installer.download_whisper_model(registry, model_key)

        self._runtime_installer = RuntimeInstaller(on_progress=self._report_install_progress)
        downloaded = False
        try:
            if not dialogs.confirm(title, message):
                return
            print(f"📦 Downloading the [{model_key}] model...")
            self._show_install_progress(title)
            download(self._runtime_installer)
            downloaded = True
        except Exception as e:
            self._close_install_progress()
            if self._install_was_cancelled():
                print("ℹ️ Download cancelled, already downloaded files are kept for the next attempt")
                return
            self.logger.error(f"Failed to download model {model_key}: {e}")
            print(f"❌ Failed to download the [{model_key}] model: {e}")
            dialogs.show_error(title, f"Download failed:\n\n{str(e)[:600]}")
            return
        finally:
            self._close_install_progress()
            self._end_engine_switch(refresh_ui=not downloaded)

        self.system_tray.refresh_menu()
        self.request_model_change(model_key)
        self._update_ui_state(self.get_current_state())

    def _switch_engine_family(self, model_key: str) -> bool:
        runtimes = self.get_runtimes()
        if self._is_onnx_model(model_key):
            gpu_runtime = self._installable_onnx_gpu_runtime(runtimes)
            if gpu_runtime:
                threading.Thread(target=self._offer_onnx_gpu_runtime, args=(gpu_runtime, model_key), daemon=True).start()
                return True
            runtime = best_onnx_runtime(runtimes, self.config_manager.get_setting('whisper', 'onnx_runtime'))
            compute_type = None
        else:
            runtime_key = self.config_manager.get_setting('whisper', 'runtime')
            runtime = next((candidate for candidate in runtimes if candidate.key == runtime_key and candidate.available), None)
            runtime = runtime or next((candidate for candidate in runtimes if candidate.key == CPU and candidate.available), None)
            compute_type = self.config_manager.get_setting('whisper', 'compute_type')
        if runtime is None:
            print(f"❌ No runtime can run the [{model_key}] model")
            return False
        return self.request_runtime_change(runtime.key, compute_type, model_key=model_key)

    def _installable_onnx_gpu_runtime(self, runtimes: list) -> Optional[Runtime]:
        if self.config_manager.get_setting('onboarding', 'onnx_gpu') == 'declined':
            return None
        onnx_runtimes = {runtime.key: runtime for runtime in runtimes if runtime.engine_type == ONNX_ASR}
        if any(onnx_runtimes[key].available for key in (ONNX_CUDA, ONNX_DIRECTML) if key in onnx_runtimes):
            return None
        return next((onnx_runtimes[key] for key in (ONNX_CUDA, ONNX_DIRECTML)
                     if key in onnx_runtimes and onnx_runtimes[key].state == INSTALLABLE), None)

    def _offer_onnx_gpu_runtime(self, runtime: Runtime, model_key: str):
        message = (f"ONNX models can run on your GPU with {runtime.label}.\n\n"
                   f"Install it now? (download {runtime.install_size}, Whisper Key restarts afterwards)\n"
                   f"No: use the CPU. You can change this later in the tray Runtime menu.")
        if not dialogs.confirm(f"Use the GPU for {model_key}", message):
            self.config_manager.update_user_setting('onboarding', 'onnx_gpu', 'declined')
            self._switch_engine_family(model_key)
            return
        if not self._begin_engine_switch():
            print("⏳ A runtime is already being installed...")
            return
        if not self._install_and_switch_runtime(runtime, model_key=model_key):
            print("ℹ️ Using the CPU for ONNX models")
            self.config_manager.update_user_setting('onboarding', 'onnx_gpu', 'declined')
            self._switch_engine_family(model_key)

    def get_runtimes(self) -> list:
        with self._runtimes_lock:
            if self._runtimes is None:
                from .whisper_cpp_engine import find_whisper_cli
                installed_binary, _ = whisper_cpp_runtime_paths()
                whisper_cpp_binary = (self.config_manager.get_setting('whisper', 'cpp_binary')
                                      or installed_binary or find_whisper_cli())
                try:
                    self._runtimes = detect_runtimes(whisper_cpp_binary)
                except Exception as e:
                    self.logger.error(f"Runtime detection failed: {e}")
                    self._runtimes = []
            return self._runtimes

    def refresh_runtimes(self):
        with self._runtimes_lock:
            self._runtimes = None

    def get_current_runtime(self) -> Optional[Runtime]:
        runtimes = self.get_runtimes()
        if self._runtime_key is None:
            engine = self.whisper_engine
            if engine.ENGINE_TYPE == ONNX_ASR:
                self._runtime_key = engine.onnx_runtime
            else:
                self._runtime_key = current_runtime_key(engine.ENGINE_TYPE, getattr(engine, 'device', 'cpu'), runtimes)
        return next((runtime for runtime in runtimes if runtime.key == self._runtime_key), None)

    def get_menu_runtimes(self) -> list:
        return runtimes_for_engine_family(self.get_runtimes(), self._is_onnx_engine())

    def get_current_compute_type(self) -> Optional[str]:
        if self.whisper_engine.ENGINE_TYPE != FASTER_WHISPER:
            return None
        return self.whisper_engine.compute_type

    def request_compute_type_change(self, compute_type: str) -> bool:
        current_runtime = self.get_current_runtime()
        return bool(current_runtime) and self.request_runtime_change(current_runtime.key, compute_type)

    def request_runtime_change(self, runtime_key: str, compute_type: Optional[str] = None, model_key: Optional[str] = None) -> bool:
        runtime = next((runtime for runtime in self.get_runtimes() if runtime.key == runtime_key), None)
        if not runtime or runtime.state == UNSUPPORTED:
            return False

        if runtime.state == INSTALLABLE:
            if not self._begin_engine_switch():
                print("⏳ A runtime is already being installed...")
                return False
            threading.Thread(target=self._install_and_switch_runtime, args=(runtime,), daemon=True).start()
            return True

        if runtime.needs_restart:
            threading.Thread(target=self._confirm_restart_into_runtime, args=(runtime,), daemon=True).start()
            return True

        compute_type = choose_compute_type(runtime, compute_type or self.get_current_compute_type())
        current_runtime = self.get_current_runtime()
        model_changes = model_key is not None and model_key != self.whisper_engine.model_key
        if current_runtime and current_runtime.key == runtime.key and compute_type == self.get_current_compute_type() and not model_changes:
            return True

        if self.get_current_state() == "recording":
            print(f"🎤 Cancelling recording to switch to [{runtime.label}]...")
            self.cancel_active_recording()

        with self._state_lock:
            if self.is_model_loading:
                print("⏳ Model already loading, please wait...")
                return False
            if self.is_processing or self.audio_recorder.get_recording_status():
                print("⏳ Busy transcribing, change the runtime again in a moment...")
                return False
            self.is_model_loading = True

        self._update_ui_state("processing")
        threading.Thread(target=self._execute_runtime_change, args=(runtime, compute_type, model_key), daemon=True).start()
        return True

    def _execute_runtime_change(self, runtime: Runtime, compute_type: Optional[str], model_key: Optional[str] = None):
        description = runtime.label if runtime.engine_type != FASTER_WHISPER else f"{runtime.label}, {compute_type}"
        if model_key:
            description = f"{model_key} on {description}"
        print(f"🔄 Switching runtime to [{description}]...")

        old_engine = self.whisper_engine
        old_runtime = self.get_current_runtime()
        old_compute_type = self.get_current_compute_type()
        changes_engine_family = (runtime.engine_type == ONNX_ASR) != (old_engine.ENGINE_TYPE == ONNX_ASR)
        whisper_config = with_whisper_cpp_paths(self.config_manager.get_whisper_config())
        whisper_config['model'] = model_key or old_engine.model_key
        if runtime.engine_type == FASTER_WHISPER:
            whisper_config['device'] = runtime.device
            whisper_config['compute_type'] = compute_type
        if runtime.engine_type == ONNX_ASR:
            whisper_config['onnx_runtime'] = runtime.key

        release_first = getattr(old_engine, 'device', 'cpu') != "cpu" or runtime.device != "cpu"
        if release_first:
            old_engine.unload()
            gc.collect()

        new_engine = None
        try:
            new_engine = create_whisper_engine(runtime.engine_type, whisper_config, self.vad_manager, old_engine.registry)
            new_engine.warm_up()
        except Exception as e:
            self.logger.error(f"Failed to switch runtime to {description}: {e}")
            print(f"❌ Failed to switch runtime: {e}")
            if new_engine:
                new_engine.close()
            new_engine = None
            gc.collect()
            if release_first:
                self._restore_engine(old_engine)
        else:
            with self._state_lock:
                self.whisper_engine = new_engine
                self._runtime_key = runtime.key
                retry_allowed = old_runtime and old_runtime.key != self._failed_runtime_key and not changes_engine_family
                self._fallback_runtime = (old_runtime, old_compute_type) if retry_allowed else None
            old_engine.unload()
            del old_engine
            gc.collect()
            self._persist_runtime(runtime, compute_type)
            if model_key:
                self.config_manager.update_user_setting('whisper', 'model', model_key)
            print(f"✅ Now transcribing with [{description}]")
        finally:
            self.set_model_loading(False)
            self.system_tray.refresh_menu()

    def offer_whisper_server_upgrade(self):
        from .whisper_cpp_engine import find_whisper_server
        installed_binary, _ = whisper_cpp_runtime_paths()
        current_runtime = self.get_current_runtime()
        if not installed_binary or find_whisper_server(installed_binary) or not current_runtime or current_runtime.key != VULKAN:
            return
        if self.config_manager.get_setting('onboarding', 'whisper_server_upgrade') == 'declined':
            return
        threading.Thread(target=self._offer_whisper_server_rebuild, args=(current_runtime,), daemon=True).start()

    def _offer_whisper_server_rebuild(self, runtime: Runtime):
        option = self._choose_install_option(runtime, upgrade=True)
        if option is None or not self._begin_engine_switch():
            return
        self._install_and_switch_runtime(runtime, upgrade=True, option=option)

    def _choose_install_option(self, runtime: Runtime, upgrade: bool = False):
        from .runtime_installer import install_options
        options = install_options(runtime.key)
        if upgrade:
            if not options:
                return None
            if dialogs.confirm(f"Upgrade {runtime.label}", (
                    f"The installed {runtime.label} runtime predates whisper-server, so the model is reloaded for every recording.\n\n"
                    f"Rebuild it now to keep the model loaded?\nDownload: {options[0].download_size}")):
                return options[0]
            self.config_manager.update_user_setting('onboarding', 'whisper_server_upgrade', 'declined')
            print("ℹ️ whisper-server rebuild skipped, this prompt will not be shown again")
            return None
        title = f"Install {runtime.label}"
        restart_note = "" if runtime.key == VULKAN else "\n\nWhisper Key restarts afterwards to use it."
        if len(options) >= 2:
            first, second = options[0], options[1]
            choice = dialogs.choose(title, (
                f"{runtime.label} is not installed yet.\n\n"
                f"Yes: {first.description} (download {first.download_size})\n"
                f"No: {second.description} (download {second.download_size})"
                f"{restart_note}"
            ))
            if choice is None:
                return None
            return first if choice else second
        if options and dialogs.confirm(title, f"{options[0].description}.\nDownload: {options[0].download_size}{restart_note}"):
            return options[0]
        return None

    def _report_install_progress(self, message: str):
        print(f"   {message}")
        self.system_tray.set_status_text(message)
        window = self._install_progress_window
        if window:
            window.set_status(message)

    def _show_install_progress(self, title: str):
        from .install_progress_window import InstallProgressWindow
        self._install_progress_window = InstallProgressWindow(title, on_cancel=self._runtime_installer.cancel)
        self._install_progress_window.show()

    def _close_install_progress(self):
        window = self._install_progress_window
        self._install_progress_window = None
        if window:
            window.close()

    def _install_was_cancelled(self) -> bool:
        installer = self._runtime_installer
        return bool(installer and installer.cancelled)

    def _install_and_switch_runtime(self, runtime: Runtime, upgrade: bool = False, model_key: Optional[str] = None,
                                    option=None) -> bool:
        from .runtime_installer import RuntimeInstaller, install_options

        self._runtime_installer = RuntimeInstaller(on_progress=self._report_install_progress)
        installed = False
        try:
            if option is None:
                option = install_options(runtime.key)[0] if model_key else self._choose_install_option(runtime, upgrade)
            if option is None:
                return False
            print(f"📦 Installing [{runtime.label}]: {option.description}")
            self._show_install_progress(f"Installing {runtime.label}")
            self._runtime_installer.install(runtime.key, option, model_key=self.whisper_engine.model_key)
            installed = True
        except Exception as e:
            self._close_install_progress()
            if self._install_was_cancelled():
                print(f"ℹ️ {runtime.label} installation cancelled")
                return False
            self.logger.error(f"Failed to install runtime {runtime.key}: {e}")
            print(f"❌ Failed to install {runtime.label}: {e}")
            dialogs.show_error(f"Install {runtime.label}", f"Installation failed:\n\n{str(e)[:600]}\n\nDetails are in the log file.")
            return False
        finally:
            self._close_install_progress()
            self._end_engine_switch(refresh_ui=not installed)

        print(f"✅ {runtime.label} installed")
        try:
            self.refresh_runtimes()
            installed_runtime = next(candidate for candidate in self.get_runtimes() if candidate.key == runtime.key)
            current_runtime = self.get_current_runtime()
            if model_key:
                self.config_manager.update_user_setting('whisper', 'model', model_key)
            if current_runtime and current_runtime.key == installed_runtime.key:
                self._reload_current_engine()
            elif installed_runtime.needs_restart:
                self._restart_into_runtime(installed_runtime)
            else:
                self.system_tray.refresh_menu()
                self.request_runtime_change(installed_runtime.key, model_key=model_key)
        finally:
            self._update_ui_state(self.get_current_state())
        return True

    def _confirm_restart_into_runtime(self, runtime: Runtime):
        if dialogs.confirm(f"Switch to {runtime.label}", f"Whisper Key restarts to switch to {runtime.label}."):
            self._restart_into_runtime(runtime)

    def _restart_into_runtime(self, runtime: Runtime):
        with self._state_lock:
            busy = self.is_model_loading or self.is_processing or self.audio_recorder.get_recording_status()
            if not busy:
                self.is_model_loading = True
        if busy:
            message = f"{runtime.label} is installed. Select it again in the Runtime menu once the current recording is done."
            print(f"⏳ {message}")
            dialogs.confirm(f"Switch to {runtime.label}", message)
            return
        self._update_ui_state("processing")
        if runtime.engine_type == ONNX_ASR:
            self.config_manager.update_user_setting('whisper', 'onnx_runtime', runtime.key)
        else:
            self.config_manager.update_user_setting('whisper', 'runtime', runtime.key)
            self.config_manager.update_user_setting('whisper', 'engine_type', runtime.engine_type)
            self.config_manager.update_user_setting('whisper', 'device', runtime.device)
            self.config_manager.update_user_setting('whisper', 'compute_type', choose_compute_type(runtime, None))
        print(f"🔄 Restarting Whisper Key to switch to [{runtime.label}]...")
        from .utils import restart_app
        try:
            restart_app()
        except Exception as e:
            self.logger.error(f"Failed to restart into {runtime.label}: {e}")
            print(f"❌ Failed to restart, restart Whisper Key manually: {e}")
            self.set_model_loading(False)

    def _reload_current_engine(self):
        with self._state_lock:
            busy = self.is_model_loading or self.is_processing or self.audio_recorder.get_recording_status()
            if not busy:
                self.is_model_loading = True
        if busy:
            print("⏳ Restart Whisper Key or switch the runtime again to use the upgraded runtime")
            return
        self._update_ui_state("processing")
        try:
            self.whisper_engine.reload()
        except Exception as e:
            self.logger.error(f"Failed to reload the upgraded runtime: {e}")
            print(f"❌ Failed to reload the upgraded runtime, restart Whisper Key: {e}")
        finally:
            self.set_model_loading(False)
            self.system_tray.refresh_menu()

    def _restore_engine(self, engine):
        print("🔄 Restoring previous runtime...")
        try:
            engine.reload()
        except Exception as e:
            self.logger.error(f"Failed to restore previous runtime: {e}")
            print(f"❌ Failed to restore previous runtime, restart Whisper Key: {e}")

    def _persist_runtime(self, runtime: Runtime, compute_type: Optional[str]):
        try:
            if runtime.engine_type == ONNX_ASR:
                self.config_manager.update_user_setting('whisper', 'onnx_runtime', runtime.key)
                return
            self.config_manager.update_user_setting('whisper', 'runtime', runtime.key)
            if runtime.engine_type == FASTER_WHISPER:
                self.config_manager.update_user_setting('whisper', 'device', runtime.device)
                self.config_manager.update_user_setting('whisper', 'compute_type', compute_type)
            self.config_manager.update_user_setting('whisper', 'engine_type', runtime.engine_type)
        except Exception as e:
            self.logger.error(f"Failed to save runtime settings: {e}")
            print(f"⚠️ Runtime switched but settings could not be saved: {e}")

    def update_transcription_mode(self, value):
        self.config_manager.update_user_setting('clipboard', 'auto_paste', value)
        self.clipboard_manager.update_auto_paste(value)

    def update_mute_output_while_recording(self, enabled: bool):
        self.output_audio_control.set_mute_output_enabled(enabled)
        self.config_manager.update_user_setting('output_audio', 'mute_while_recording', enabled)

    def update_pause_media_while_recording(self, enabled: bool):
        self.output_audio_control.set_pause_media_enabled(enabled)
        self.config_manager.update_user_setting('output_audio', 'pause_media_while_recording', enabled)

    def update_audio_feedback(self, enabled: bool):
        self.audio_feedback.set_enabled(enabled)
        self.config_manager.update_user_setting('audio_feedback', 'enabled', enabled)

    def update_copy_to_clipboard(self, value):
        self.clipboard_manager.update_copy_to_clipboard(value)
        self.config_manager.update_user_setting('clipboard', 'copy_to_clipboard', value)

    def _execute_model_change(self, new_model_key: str):
        def progress_callback(message: str):
            if "ready" in message.lower() or "already loaded" in message.lower():
                print(f"✅ Successfully switched to [{new_model_key}] model")
                self.config_manager.update_user_setting('whisper', 'model', new_model_key)
                self.set_model_loading(False)
            elif "failed" in message.lower():
                print(f"❌ Failed to change model: {message}")
                self.set_model_loading(False)
            else:
                print(f"🔄 {message}")
                self.set_model_loading(True)
        
        self._fallback_runtime = None
        try:
            print(f"🔄 Switching to [{new_model_key}] model...")
            
            self.whisper_engine.change_model(new_model_key, progress_callback)
            
        except Exception as e:
            self.logger.error(f"Failed to initiate model change: {e}")
            print(f"❌ Failed to change model: {e}")
            self.set_model_loading(False)

    def get_available_audio_devices(self, host_filter: Optional[str] = None):
        host_name = host_filter if host_filter is not None else self._current_audio_host
        return AudioRecorder.get_available_audio_devices(host_name)

    def get_current_audio_device_id(self):
        return self.audio_recorder.get_device_id()

    def get_available_audio_hosts(self):
        try:
            hostapis = sd.query_hostapis()
            devices = sd.query_devices()
        except Exception as e:
            self.logger.error(f"Failed to query audio hosts: {e}")
            return []

        hosts_with_input = {}
        for index, host in enumerate(hostapis):
            hosts_with_input[index] = {
                'name': host['name'],
                'index': index,
                'has_input': False
            }

        for device in devices:
            if device.get('max_input_channels', 0) > 0:
                host_index = device['hostapi']
                if host_index in hosts_with_input:
                    hosts_with_input[host_index]['has_input'] = True

        return [
            {'name': host['name'], 'index': host['index']}
            for host in hosts_with_input.values()
            if host['has_input']
        ]

    def get_current_audio_host(self) -> Optional[str]:
        return self._current_audio_host

    def set_audio_host(self, host_name: str) -> bool:
        if not host_name:
            return False

        available_hosts = self.get_available_audio_hosts()
        normalized_lookup = {host['name'].lower(): host for host in available_hosts}
        host_entry = normalized_lookup.get(host_name.lower())

        if not host_entry:
            self.logger.warning(f"Requested audio host '{host_name}' is not available")
            return False

        canonical_name = host_entry['name']
        if canonical_name == self._current_audio_host:
            return True

        self._current_audio_host = canonical_name
        self.config_manager.update_audio_host(canonical_name)
        self.logger.info(f"Audio host changed to {canonical_name}")

        self._ensure_audio_device_for_host(canonical_name)
        self.system_tray.refresh_menu()
        return True

    def request_audio_device_change(self, device_id: int, device_name: str):
        current_state = self.get_current_state()

        if device_id == self.audio_recorder.device:
            return True

        if current_state == "recording":
            print(f"🎤 Cancelling recording to switch audio device...")
            self.cancel_active_recording()
            self._execute_audio_device_change(device_id, device_name)
            return True

        if current_state == "processing":
            print(f"⏳ Queueing audio device change until transcription completes...")
            self._pending_device_change = (device_id, device_name)
            return True

        if current_state == "idle":
            self._execute_audio_device_change(device_id, device_name)
            return True

        self.logger.warning(f"Unexpected state for device change: {current_state}")
        return False

    def _execute_audio_device_change(self, device_id: int, device_name: str):
        try:
            print(f"🎤 Switching to: {device_name}")

            channels = self.audio_recorder.channels
            dtype = self.audio_recorder.dtype
            max_duration = self.audio_recorder.max_duration
            on_max_duration = self.audio_recorder.on_max_duration_reached
            vad_manager = self.audio_recorder.vad_manager
            streaming_manager = self.audio_recorder.streaming_manager
            on_streaming_result = self.audio_recorder.on_streaming_result

            new_recorder = AudioRecorder(
                on_vad_event=self.handle_vad_event,
                channels=channels,
                dtype=dtype,
                max_duration=max_duration,
                on_max_duration_reached=on_max_duration,
                vad_manager=vad_manager,
                streaming_manager=streaming_manager,
                on_streaming_result=on_streaming_result,
                on_vad_probability=self.handle_vad_probability,
                device=device_id if device_id != -1 else None,
                on_monitoring_failed=self.handle_monitoring_failed
            )

            previous_recorder = self.audio_recorder
            self.audio_recorder = new_recorder
            previous_recorder.stop_monitoring()
            previous_recorder.cancel_recording()

            print(f"✅ Successfully switched audio device to: {device_name}")

        except Exception as e:
            self.logger.error(f"Failed to change audio device: {e}")
            print(f"❌ Failed to switch audio device: {e}")

        self._apply_auto_trigger()

    def _initialize_audio_host(self):
        try:
            configured_host = self.config_manager.get_setting('audio', 'host')
        except KeyError:
            configured_host = None

        available_hosts = self.get_available_audio_hosts()
        resolved_host = self._resolve_audio_host(configured_host, available_hosts)

        self._current_audio_host = resolved_host

        if resolved_host != configured_host:
            self.config_manager.update_audio_host(resolved_host)

    def _resolve_audio_host(self, configured_host: Optional[str], available_hosts):
        if not available_hosts:
            return None

        normalized_lookup = {
            host['name'].lower(): host['name']
            for host in available_hosts
        }

        if configured_host:
            match = normalized_lookup.get(configured_host.lower())
            if match:
                return match

        preferred_host = self._preferred_platform_host()
        if preferred_host:
            preferred_match = normalized_lookup.get(preferred_host.lower())
            if preferred_match:
                return preferred_match

        return available_hosts[0]['name']

    def _preferred_platform_host(self) -> Optional[str]:
        system_name = platform.system().lower()
        if system_name == 'windows':
            return 'WASAPI'
        return None

    def _ensure_audio_device_for_host(self, host_name: Optional[str]):
        if not host_name or not self.audio_recorder:
            return

        try:
            current_device_id = self.audio_recorder.get_device_id()
        except Exception as e:
            self.logger.error(f"Unable to read current audio device: {e}")
            return

        if self._device_matches_host(current_device_id, host_name):
            return

        fallback_device_id = self._get_default_device_for_host(host_name)
        if fallback_device_id is None:
            self.logger.warning(f"No input devices available for host {host_name}")
            return

        device_name = self._get_device_name(fallback_device_id)
        success = self.request_audio_device_change(fallback_device_id, device_name)

        if not success:
            self.logger.warning(f"Failed to switch to fallback device {fallback_device_id} for host {host_name}")

    def _device_matches_host(self, device_id: int, host_name: str) -> bool:
        try:
            device_info = sd.query_devices(device_id)
            host_info = sd.query_hostapis(device_info['hostapi'])
            return host_info['name'].lower() == host_name.lower()
        except Exception:
            return False

    def _get_default_device_for_host(self, host_name: str) -> Optional[int]:
        try:
            target_index = None
            target_host = None
            hostapis = sd.query_hostapis()
            for idx, host in enumerate(hostapis):
                if host['name'].lower() == host_name.lower():
                    target_index = idx
                    target_host = host
                    break
            else:
                return None

            default_input = target_host.get('default_input_device', -1)
            if default_input is not None and default_input >= 0:
                device_info = sd.query_devices(default_input)
                if device_info.get('max_input_channels', 0) > 0:
                    return default_input

            all_devices = sd.query_devices()
            for idx, device in enumerate(all_devices):
                if device['hostapi'] == target_index and device.get('max_input_channels', 0) > 0:
                    return idx
        except Exception as e:
            self.logger.error(f"Failed to determine default device for host {host_name}: {e}")

        return None

    def _get_device_name(self, device_id: int) -> str:
        try:
            device_info = sd.query_devices(device_id)
            return device_info.get('name', f"Device {device_id}")
        except Exception:
            return f"Device {device_id}"

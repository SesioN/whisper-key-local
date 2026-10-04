#!/usr/bin/env python3

from .utils import setup_portaudio_path, setup_nvidia_dll_path
setup_portaudio_path()
setup_nvidia_dll_path()

from .runtime_loader import ROCM, VULKAN, activate_selected_runtime, runtime_activation_warning
activate_selected_runtime()

import argparse
import logging
import os
import signal
import sys
import threading

from .platform import app, permissions, console, IS_WINDOWS
from .config_manager import ConfigManager
from .audio_recorder import AudioRecorder
from .hotkey_listener import HotkeyListener
from .whisper_engine import create_whisper_engine
from .runtime_options import apply_runtime_selection
from .voice_activity_detection import VadManager
from .clipboard_manager import ClipboardManager
from .state_manager import StateManager
from .terminal_title import TerminalTitle
from .text_postprocessor import TextPostProcessor
from .system_tray import SystemTray
from .audio_feedback import AudioFeedback
from .instance_manager import guard_against_multiple_instances
from .model_registry import ModelRegistry
from .streaming_manager import StreamingManager
from .voice_commands import VoiceCommandManager
from .hardware_detection import detect_and_print as detect_hardware
from .onboarding import check_gpu
from .update_checker import check_for_updates
from .utils import get_user_app_data_path, get_version, OptionalComponent

def setup_logging(config_manager: ConfigManager):
    log_config = config_manager.get_logging_config()
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
    
    root_logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_config['file']['enabled']:
        whisperkey_dir = get_user_app_data_path()
        log_file_path = os.path.join(whisperkey_dir, log_config['file']['filename'])
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_config['level']))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    if log_config['console']['enabled']:
        console_handler = logging.StreamHandler()
        console_level = log_config['console'].get('level', 'WARNING')
        console_handler.setLevel(getattr(logging, console_level))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

def setup_exception_handler():
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logging.getLogger().error("Uncaught exception", 
                                 exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = exception_handler

def setup_audio_recorder(audio_config, state_manager, vad_manager, streaming_manager):
    return AudioRecorder(
        channels=audio_config['channels'],
        dtype=audio_config['dtype'],
        max_duration=audio_config['max_duration'],
        on_max_duration_reached=state_manager.handle_max_recording_duration_reached,
        on_vad_event=state_manager.handle_vad_event,
        vad_manager=vad_manager,
        streaming_manager=streaming_manager,
        on_streaming_result=state_manager.handle_streaming_result,
        on_vad_probability=state_manager.handle_vad_probability,
        device=audio_config['input_device'],
        on_monitoring_failed=state_manager.handle_monitoring_failed
    )

def setup_vad(vad_config):
    return VadManager(
        vad_precheck_enabled=vad_config['vad_precheck_enabled'],
        vad_realtime_enabled=vad_config['vad_realtime_enabled'],
        vad_onset_threshold=vad_config['vad_onset_threshold'],
        vad_offset_threshold=vad_config['vad_offset_threshold'],
        vad_min_speech_duration=vad_config['vad_min_speech_duration'],
        vad_silence_timeout_seconds=vad_config['vad_silence_timeout_seconds'],
        auto_trigger_silence_seconds=vad_config['auto_trigger_silence_seconds']
    )

def setup_vad_sensitivity_window(vad_config, state_manager):
    if not IS_WINDOWS:
        return None
    try:
        from .vad_sensitivity_window import VadSensitivityWindow
    except ImportError as e:
        logging.getLogger(__name__).warning(f"VAD sensitivity window not available: {e}")
        return None
    return VadSensitivityWindow(
        onset_threshold=vad_config['vad_onset_threshold'],
        on_threshold_selected=state_manager.update_vad_onset_threshold,
        on_opened=state_manager.handle_vad_sensitivity_window_opened,
        on_closed=state_manager.handle_vad_sensitivity_window_closed
    )

def setup_streaming(streaming_config, model_registry):
    return StreamingManager(
        streaming_enabled=streaming_config.get('streaming_enabled', False),
        streaming_model=streaming_config.get('streaming_model', 'standard'),
        model_registry=model_registry
    )

def setup_whisper_engine(whisper_config, vad_manager, model_registry, config_manager=None, loading_screen=None):
    whisper_config = apply_runtime_selection(whisper_config)
    engine_type = whisper_config['engine_type']
    try:
        return create_whisper_engine(
            engine_type=engine_type,
            whisper_config=whisper_config,
            vad_manager=vad_manager,
            model_registry=model_registry,
        )
    except (RuntimeError, OSError) as e:
        if engine_type == 'whisper_cpp':
            return _handle_whisper_cpp_failure(e, whisper_config, vad_manager, model_registry)
        if isinstance(e, RuntimeError) and whisper_config.get('device') == 'cuda' and config_manager:
            if loading_screen:
                loading_screen.close()
            return _handle_gpu_failure(e, whisper_config, vad_manager, model_registry, config_manager)
        raise

def setup_terminal_title(terminal_title_config):
    return TerminalTitle(frames_config=terminal_title_config)

def setup_text_postprocessor(post_processing_config):
    return TextPostProcessor(
        strip_trailing_period=post_processing_config.get('strip_trailing_period', False),
        corrections=post_processing_config.get('corrections') or {}
    )

def setup_clipboard_manager(clipboard_config):
    return ClipboardManager(
        auto_paste=clipboard_config['auto_paste'],
        delivery_method=clipboard_config['delivery_method'],
        paste_hotkey=clipboard_config['paste_hotkey'],
        paste_pre_paste_delay=clipboard_config['paste_pre_paste_delay'],
        paste_preserve_clipboard=clipboard_config['paste_preserve_clipboard'],
        paste_clipboard_restore_delay=clipboard_config['paste_clipboard_restore_delay'],
        copy_to_clipboard=clipboard_config['copy_to_clipboard'],
        type_auto_enter_delay=clipboard_config['type_auto_enter_delay'],
        type_auto_enter_delay_per_100_chars=clipboard_config['type_auto_enter_delay_per_100_chars'],
        macos_key_simulation_delay=clipboard_config['macos_key_simulation_delay']
    )

def setup_audio_feedback(audio_feedback_config):
    return AudioFeedback(
        enabled=audio_feedback_config['enabled'],
        transcription_complete_enabled=audio_feedback_config['transcription_complete_enabled'],
        ready_enabled=audio_feedback_config['ready_enabled'],
        start_sound=audio_feedback_config['start_sound'],
        stop_sound=audio_feedback_config['stop_sound'],
        cancel_sound=audio_feedback_config['cancel_sound'],
        transcription_complete_sound=audio_feedback_config['transcription_complete_sound'],
        ready_sound=audio_feedback_config['ready_sound']
    )

def setup_voice_commands(voice_commands_config, clipboard_manager, log_transcriptions=False):
    return VoiceCommandManager(
        enabled=voice_commands_config['enabled'],
        clipboard_manager=clipboard_manager,
        log_transcriptions=log_transcriptions
    )

def setup_loading_screen(loading_screen_config):
    # Windows only: Tk on macOS must run on the main thread, which the app event loop owns
    if not IS_WINDOWS or not loading_screen_config.get('enabled', False):
        return None
    try:
        from .loading_screen import LoadingScreen
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Loading screen not available: {e}")
        return None
    return LoadingScreen(version=get_version())

def setup_system_tray(tray_config, config_manager, state_manager, model_registry, console_config=None):
    return SystemTray(
        state_manager=state_manager,
        tray_config=tray_config,
        config_manager=config_manager,
        model_registry=model_registry,
        console_config=console_config
    )

def setup_floating_widget(floating_widget_config, state_manager):
    if not IS_WINDOWS:
        return None
    try:
        from .floating_widget import FloatingWidget
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Floating widget not available: {e}")
        return None
    return FloatingWidget(
        on_click=state_manager.toggle_recording,
        on_position_changed=state_manager.save_floating_widget_position,
        on_mute_click=state_manager.toggle_mute,
        on_lock_changed=state_manager.save_floating_widget_locked,
        size=floating_widget_config['size'],
        save_position=floating_widget_config['save_position'],
        position=floating_widget_config['position'],
        locked=floating_widget_config['locked']
    )

def run_gpu_onboarding(config_manager, whisper_config):
    gpu_status = config_manager.config.get('onboarding', {}).get('gpu', 'pending')
    if gpu_status != 'pending' or whisper_config.get('runtime') == VULKAN:
        return whisper_config, None

    gpu_class, gpu_name, ct2_works = detect_hardware(whisper_config['device'])

    if gpu_class and gpu_class.startswith('amd') and not ct2_works:
        from .terminal_ui import BOLD_GREEN, RESET, prompt_choice

        INSTALL_ROCM = 1
        USE_WHISPER_CPP = 2
        CPU_ONLY = 4

        choice = prompt_choice(
            "GPU acceleration available",
            [
                ("Use AMD ROCm (faster_whisper)", "Installs the ROCm runtime after startup"),
                ("Use whisper.cpp with Vulkan", "Installs the Vulkan runtime after startup"),
                ("Skip for now", "Use CPU this session"),
                ("Use CPU only", "Don't ask again"),
            ],
            subtitle=f"Detected {gpu_name}. Choose transcription engine:",
        )
        print()

        if choice == INSTALL_ROCM and gpu_class == 'amd_rdna1':
            from .onboarding import install_gpu_runtime
            install_gpu_runtime(gpu_class, gpu_name, config_manager)
            return config_manager.get_whisper_config(), None

        if choice in (INSTALL_ROCM, USE_WHISPER_CPP):
            config_manager.update_user_setting('onboarding', 'gpu_class', gpu_class)
            config_manager.update_user_setting('onboarding', 'gpu', 'complete')
            print(f"{BOLD_GREEN}The runtime is set up after startup. You can change it later in the tray Runtime menu.{RESET}\n")
            return whisper_config, ROCM if choice == INSTALL_ROCM else VULKAN

        if choice == CPU_ONLY:
            config_manager.update_user_setting('whisper', 'device', 'cpu')
            config_manager.update_user_setting('whisper', 'compute_type', 'int8')
            config_manager.update_user_setting('whisper', 'runtime', 'cpu')
            config_manager.update_user_setting('onboarding', 'gpu_class', gpu_class)
            config_manager.update_user_setting('onboarding', 'gpu', 'skipped')

        return whisper_config, None

    check_gpu(gpu_class, gpu_name, ct2_works, whisper_config['device'], config_manager)
    return config_manager.get_whisper_config(), None


def _handle_whisper_cpp_failure(error, whisper_config, vad_manager, model_registry):
    logging.getLogger(__name__).error(f"whisper.cpp engine failed to start: {error}")
    print(f"\n❌ whisper.cpp engine unavailable: {error}")
    print("   Falling back to faster-whisper on CPU for this session.\n")
    fallback_config = {**whisper_config, 'device': 'cpu', 'compute_type': 'int8'}
    return create_whisper_engine('faster_whisper', fallback_config, vad_manager, model_registry)


def _handle_gpu_failure(error, whisper_config, vad_manager, model_registry, config_manager):
    from .onboarding import handle_gpu_failure
    handle_gpu_failure(error, config_manager)
    whisper_config['runtime'] = 'cpu'
    whisper_config['device'] = 'cpu'
    whisper_config['compute_type'] = 'int8'
    return setup_whisper_engine(whisper_config, vad_manager, model_registry)


def setup_signal_handlers(shutdown_event):
    def signal_handler(signum, frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def setup_hotkey_listener(hotkey_config, state_manager, voice_commands_enabled=True):
    return HotkeyListener(
        state_manager=state_manager,
        recording_hotkey=hotkey_config['recording_hotkey'],
        stop_key=hotkey_config['stop_key'],
        auto_send_key=hotkey_config.get('auto_send_key'),
        cancel_combination=hotkey_config.get('cancel_combination'),
        command_hotkey=hotkey_config.get('command_hotkey') if voice_commands_enabled else None,
        recording_mode=hotkey_config.get('recording_mode', 'toggle')
    )

def shutdown_app(hotkey_listener: HotkeyListener, state_manager: StateManager, logger: logging.Logger):
    try:
        if hotkey_listener and hotkey_listener.is_active():
            logger.info("Stopping hotkey listener...")
            hotkey_listener.stop_listening()
    except Exception as ex:
        logger.error(f"Error stopping hotkey listener: {ex}")

    if state_manager:
        state_manager.shutdown()

def main():
    console.setup()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    app.setup()

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run as separate test instance')
    args = parser.parse_args()

    instance_name = "WhisperKeyLocal_test" if args.test else "WhisperKeyLocal"
    mutex_handle = guard_against_multiple_instances(instance_name)

    mode_label = " [TEST]" if args.test else ""
    print(f"Starting Whisper Key [{get_version()}]{mode_label}...")
    
    shutdown_event = threading.Event()
    setup_signal_handlers(shutdown_event)
    
    hotkey_listener = None
    state_manager = None
    logger = None
    loading_screen = OptionalComponent(None)
    
    try:
        config_manager = ConfigManager()
        terminal_title = setup_terminal_title(config_manager.get_terminal_title_config())
        setup_logging(config_manager)
        logger = logging.getLogger(__name__)
        setup_exception_handler()

        check_for_updates(config_manager, test_mode=args.test)

        whisper_config = config_manager.get_whisper_config()
        audio_config = config_manager.get_audio_config()
        hotkey_config = config_manager.get_hotkey_config()
        clipboard_config = config_manager.get_clipboard_config()
        tray_config = config_manager.get_system_tray_config()
        audio_feedback_config = config_manager.get_audio_feedback_config()
        vad_config = config_manager.get_vad_config()
        streaming_config = config_manager.get_streaming_config()
        voice_commands_config = config_manager.get_voice_commands_config()
        post_processing_config = config_manager.get_post_processing_config()
        console_config = config_manager.get_console_config()
        floating_widget_config = config_manager.get_floating_widget_config()
        log_config = config_manager.get_logging_config()
        log_transcriptions = log_config.get('log_transcriptions', False)

        activation_warning = runtime_activation_warning()
        if activation_warning:
            logger.warning(activation_warning)
            print(f"⚠ {activation_warning}")

        whisper_config, onboarding_runtime = run_gpu_onboarding(config_manager, whisper_config)

        loading_screen = OptionalComponent(setup_loading_screen(config_manager.get_loading_screen_config()))
        loading_screen.show()

        model_registry = ModelRegistry(
            whisper_models_config=whisper_config.get('models', {}),
            streaming_models_config=streaming_config.get('models', {})
        )
        vad_manager = setup_vad(vad_config)
        streaming_manager = setup_streaming(streaming_config, model_registry)
        loading_screen.set_status("Starting whisper.cpp server..." if whisper_config.get('runtime') == VULKAN else "Loading Whisper model...")
        whisper_engine = setup_whisper_engine(whisper_config, vad_manager, model_registry, config_manager, loading_screen)
        loading_screen.set_status("Loading streaming model...")
        streaming_manager.initialize()
        loading_screen.set_status("Finishing startup...")
        clipboard_manager = setup_clipboard_manager(clipboard_config)
        audio_feedback = setup_audio_feedback(audio_feedback_config)
        voice_command_manager = setup_voice_commands(voice_commands_config, clipboard_manager, log_transcriptions)
        text_postprocessor = setup_text_postprocessor(post_processing_config)

        state_manager = StateManager(
            audio_recorder=None,
            whisper_engine=whisper_engine,
            clipboard_manager=clipboard_manager,
            system_tray=None,
            config_manager=config_manager,
            audio_feedback=audio_feedback,
            vad_manager=vad_manager,
            text_postprocessor=text_postprocessor,
            voice_command_manager=voice_command_manager,
            terminal_title=terminal_title
        )
        audio_recorder = setup_audio_recorder(audio_config, state_manager, vad_manager, streaming_manager)
        state_manager.attach_vad_sensitivity_window(setup_vad_sensitivity_window(vad_config, state_manager))
        system_tray = setup_system_tray(tray_config, config_manager, state_manager, model_registry, console_config)
        floating_widget = setup_floating_widget(floating_widget_config, state_manager)
        state_manager.attach_components(audio_recorder, system_tray, floating_widget)
        
        hotkey_listener = setup_hotkey_listener(hotkey_config, state_manager, voice_commands_config['enabled'])

        state_manager.get_runtimes()
        system_tray.start()
        if floating_widget and floating_widget_config['enabled']:
            floating_widget.show()
        terminal_title.start()
        loading_screen.close()

        if clipboard_config['auto_paste']:
            if not permissions.check_accessibility_permission():
                if not permissions.handle_missing_permission(config_manager):
                    app.run_event_loop(shutdown_event)
                    return
                clipboard_manager.update_auto_paste(False)

        print("🚀 Whisper Key ready!")
        state_manager.play_ready_sound()
        config_manager.print_startup_hotkey_instructions()
        print("   [CTRL+C] to quit", flush=True)

        system_tray.apply_console_settings()

        if onboarding_runtime and not state_manager.request_runtime_change(onboarding_runtime):
            print("⚠ The selected GPU runtime is not available on this system. Choose another one in the tray Runtime menu.")

        app.run_event_loop(shutdown_event)
            
    except KeyboardInterrupt:
        logger.info("Application shutting down...")
        print("\nShutting down application...")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error occurred: {e}")
        
    finally:
        loading_screen.close()
        shutdown_app(hotkey_listener, state_manager, logger)

if __name__ == "__main__":
    main()

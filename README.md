# Whisper Key - Local Speech-to-Text

Global hotkeys to record speech and transcribe directly to your cursor.

> This is an independent fork of [PinW/whisper-key-local](https://github.com/PinW/whisper-key-local) with its own
> releases. Bugs and ideas: [issues](https://github.com/SesioN/whisper-key-local/issues).

## ✨ Features

- **Global Hotkey**: Start recording speech from any app
- **Auto-Paste**: Transcribe directly to cursor
- **Auto-Send**: Optionally auto-send with ENTER keypress
- **Local/Offline**: Voice data never leaves your computer
- **CPU Ready**: Small, efficient models available
- **GPU Ready**: NVIDIA (CUDA), AMD (ROCm) and any Vulkan GPU via whisper.cpp; runtimes install on demand (Windows)
- **Cross-platform**: Works on Windows and macOS (GPU runtimes, floating button, loading screen and sensitivity window are Windows only)
- **Voice Commands**: Trigger shortcuts, text snippets, and shell commands by voice — [docs](docs/voice-commands.md)
- **More engines**: whisper.cpp (Vulkan) and ONNX models (Parakeet-TDT-0.6B-V3, Qwen3-ASR-1.7B), selectable from the tray
- **Hands-free**: Voice-activated recording, plus an always-on-top record button (Windows)
- **Configurable**: Customize hotkeys, models, and [much more](#️-configuration)

## 🚀 Quick Start

### Windows App (Recommended)

1. Download `whisper-key.exe` (or `whisper-key-no-term.exe` for no console window) from the
   [latest release](https://github.com/SesioN/whisper-key-local/releases/latest)
   `whisper-key-hideable.exe` is the variant for `console.start_hidden`
2. Run it. On startup the app checks this repository's releases for updates.

### Python Package

Requires Python 3.11 or newer (the Windows app uses Python 3.12). This fork is not on PyPI (`pip install whisper-key-local` installs the upstream app).
Download the `.whl` from the [latest release](https://github.com/SesioN/whisper-key-local/releases/latest), then:

```bash
pipx install ./whisper_key_local-<version>-py3-none-any.whl
```

Then run: `whisper-key` (or `wk` for short)

### From Source

```bash
git clone https://github.com/SesioN/whisper-key-local.git
cd whisper-key-local
pip install -e .
python whisper-key.py
```

## 🎤 Basic Usage

| Hotkey | Windows | macOS |
|--------|---------|-------|
| Start recording | `Ctrl+Win` | `Fn+Ctrl` |
| Stop & transcribe | `Ctrl` | `Fn` |
| Stop & auto-send | `Alt` | `Option` |
| Cancel recording | `Esc` | `Shift` |
| Voice command mode | `Alt+Win` | `Fn+Command` |

Open the system tray / menu bar icon to:
- Change transcription model, runtime (CPU / CUDA / ROCm / Vulkan / ONNX) and precision
- Select audio host and input device, or mute the microphone
- Toggle auto-paste, copy to clipboard, audio feedback and voice-activated recording
- Open the voice detection sensitivity window and floating button options (Windows)
- Manage shortcuts: record a primary and secondary binding per action, or unset one (Windows, **Shortcuts...**)
- Manage voice commands: add, edit, test and switch them on or off (Windows, **Voice commands...**)
- Open the settings, commands and log files

## 🗣️ Voice Commands

Speak trigger phrases to run shell commands and more. Define in:
- **Windows:** `%APPDATA%\whisperkey\commands.yaml`
- **macOS:** `~/.whisperkey/commands.yaml`

```yaml
commands:
  # Send a keyboard shortcut
  - trigger: "undo"
    hotkey: "ctrl+z"
  # Deliver pre-written text
  - trigger: "my email"
    type: "user@example.com"
  # Run a shell command
  - trigger: "open notepad"
    run: 'notepad.exe'
```

See the **[Voice Commands Guide](docs/voice-commands.md)** for full details.

## ⚡ GPU Acceleration

On Windows, Whisper Key detects your GPU on first launch and offers to install the matching runtime. Runtimes can be
installed and switched later from the tray **Runtime** menu; they are stored in `%LOCALAPPDATA%\whisperkey\runtimes`.

| Runtime | Engine | Hardware |
|---|---|---|
| CPU | faster-whisper | any |
| NVIDIA CUDA | faster-whisper | NVIDIA |
| AMD ROCm | faster-whisper | AMD |
| Vulkan | whisper.cpp (whisper-server) | GPU with a Vulkan driver |
| ONNX CPU / DirectML / CUDA | onnx-asr (Parakeet, Qwen3-ASR models) | any / DirectX 12 GPU / NVIDIA |

The **Precision** menu lists the compute types the active device supports.

For manual setup or troubleshooting, see the **[GPU Setup Guide](docs/gpu-setup.md)**.

## ⚙️ Configuration

Local settings at:
- **Windows:** `%APPDATA%\whisperkey\user_settings.yaml`
- **macOS:** `~/.whisperkey/user_settings.yaml`

Delete this file and restart app to reset to defaults.

| Option | Default | Notes |
|--------|---------|-------|
| **Whisper** |||
| `whisper.model` | `tiny` | Any model defined in `whisper.models` |
| `whisper.runtime` | `cpu` | cpu, cuda, rocm or vulkan; set from the tray Runtime menu |
| `whisper.onnx_runtime` | `onnx-cpu` | onnx-cpu, onnx-directml or onnx-cuda; used by ONNX models |
| `whisper.engine_type` | `faster_whisper` | Derived from `whisper.runtime` (vulkan selects whisper_cpp) |
| `whisper.device` | `cpu` | cpu or cuda (NVIDIA and AMD); set from the runtime — [setup guide](docs/gpu-setup.md) |
| `whisper.compute_type` | `int8` | int8, int8_float32, int8_float16, int8_bfloat16, int16, float16, bfloat16, float32 (faster-whisper only) |
| `whisper.cpp_binary` | `""` | Custom whisper-cli.exe path (auto-detected if empty) |
| `whisper.cpp_model_dir` | `""` | Custom ggml model directory (auto-detected if empty) |
| `whisper.language` | `auto` | auto or language code (en, es, fr, etc.) |
| `whisper.beam_size` | `5` | Higher = more accurate but slower (1-10) |
| `whisper.initial_prompt` | `""` | Guide transcription style, language variant, or script |
| `whisper.hotwords` | `[]` | Words the model should favor (names, technical terms) |
| `whisper.models` | (see config) | Add custom HuggingFace or local models |
| **Post-Processing** |||
| `post_processing.strip_trailing_period` | `false` | Strip trailing period from output |
| `post_processing.corrections` | `{}` | Fix recurring misheard words, e.g. `CAPEX: [cap x]` |
| **Hotkeys** |||
| `hotkey.recording_hotkey` | `[ctrl+win, ""]` / `[fn+ctrl, ""]` | Start recording. Every hotkey takes `[primary, secondary]`; `""` leaves a slot unset, an action with no bindings is disabled. Edit them from the tray **Shortcuts...** window (Windows) |
| `hotkey.stop_key` | `[ctrl, ""]` / `[fn, ""]` | Stop recording |
| `hotkey.auto_send_key` | `[alt, ""]` / `[option, ""]` | Stop + paste + Enter |
| `hotkey.cancel_combination` | `[esc, ""]` / `[shift, ""]` | Cancel recording |
| `hotkey.recording_mode` | `toggle` | toggle or push_to_talk |
| `hotkey.command_hotkey` | `[alt+win, ""]` / `[fn+command, ""]` | Voice command mode |
| **Voice Activity Detection** |||
| `vad.vad_precheck_enabled` | `true` | Prevent hallucinations on silence |
| `vad.vad_onset_threshold` | `0.7` | Speech detection start (0.0-1.0) |
| `vad.vad_offset_threshold` | `0.55` | Speech detection end (0.0-1.0) |
| `vad.vad_min_speech_duration` | `0.1` | Min speech segment (seconds); also how long speech must last to start a voice-activated recording |
| `vad.vad_realtime_enabled` | `true` | Auto-stop on silence |
| `vad.vad_silence_timeout_seconds` | `30.0` | Seconds before auto-stop |
| `vad.auto_trigger_enabled` | `false` | Start recording when speech is detected, stop after a short silence. Keeps the microphone open and the VAD running (about 0.1 ms per 16 ms audio block, under 1% of one core); any nearby speech (TV, calls) is transcribed and delivered like a hotkey recording, including auto-paste unless `vad.auto_trigger_paste` is false. The tray tooltip shows when the microphone is listening. The live streaming preview starts without the 1 s pre-roll; the final transcription includes it |
| `vad.auto_trigger_silence_seconds` | `1.5` | Silence that ends a voice-activated recording |
| `vad.auto_trigger_paste` | `true` | false = voice-activated recordings are only copied to the clipboard |
| **Audio** |||
| `audio.host` | `null` | Audio API (WASAPI, Core Audio, etc.) |
| `audio.channels` | `1` | 1 = mono, 2 = stereo |
| `audio.dtype` | `float32` | float32/int16/int24/int32 |
| `audio.max_duration` | `900` | Max recording seconds (0 = unlimited) |
| `audio.input_device` | `default` | Device ID or "default" |
| **Clipboard** |||
| `clipboard.auto_paste` | `true` | false = clipboard only |
| `clipboard.copy_to_clipboard` | `false` | Keep transcription on clipboard after auto-paste |
| `clipboard.delivery_method` | `paste` | paste (Ctrl+V) or type (direct injection) |
| `clipboard.paste_hotkey` | `ctrl+v` / `cmd+v` | Paste key simulation |
| `clipboard.paste_pre_paste_delay` | `0.05` | Delay after copy, before paste hotkey (seconds) |
| `clipboard.paste_preserve_clipboard` | `true` | Restore clipboard after paste |
| `clipboard.paste_clipboard_restore_delay` | `0.5` | Delay before clipboard restore (seconds) |
| `clipboard.type_auto_enter_delay` | `0.15` | Delay before ENTER after typing (seconds) |
| `clipboard.type_auto_enter_delay_per_100_chars` | `0.1` | Extra ENTER delay per 100 typed chars (seconds) |
| **Logging** |||
| `logging.level` | `INFO` | DEBUG/INFO/WARNING/ERROR/CRITICAL |
| `logging.file.enabled` | `true` | Write to app.log |
| `logging.log_transcriptions` | `false` | Include transcribed text in log (privacy) |
| `logging.console.enabled` | `true` | Print to console |
| `logging.console.level` | `WARNING` | Console verbosity |
| **Audio Feedback** |||
| `audio_feedback.enabled` | `true` | Play sounds on record/stop (also toggled from the tray) |
| `audio_feedback.transcription_complete_enabled` | `false` | Play sound on transcription complete |
| `audio_feedback.ready_enabled` | `true` | Play sound when app finishes loading |
| `audio_feedback.start_sound` | `assets/sounds/...` | Custom sound file path |
| `audio_feedback.stop_sound` | `assets/sounds/...` | Custom sound file path |
| `audio_feedback.cancel_sound` | `assets/sounds/...` | Custom sound file path |
| `audio_feedback.transcription_complete_sound` | `assets/sounds/...` | Custom sound file path |
| `audio_feedback.ready_sound` | `assets/sounds/...` | Custom sound file path |
| **System Tray** |||
| `system_tray.enabled` | `true` | Show tray icon |
| `system_tray.tooltip` | `Whisper Key` | Hover text |
| **Floating Button** |||
| `floating_widget.enabled` | `false` | Always-on-top button to start/stop recording by mouse (Windows only) |
| `floating_widget.size` | `big` | small, medium or big |
| `floating_widget.save_position` | `false` | Restore the last dragged position on startup |
| `floating_widget.locked` | `false` | Keep the button locked in place |
| **Streaming Preview (experimental)** |||
| `streaming.streaming_enabled` | `false` | Live speech preview while recording (sherpa-onnx) |
| `streaming.streaming_model` | `zipformer.tiny.en` | Streaming model |
| **Terminal Title** |||
| `terminal_title.idle` | `""` | Tab title prefix when idle: static string or `[prefix, seconds]` animation frames |
| `terminal_title.recording` | 🔴 blink | Tab title prefix while recording |
| `terminal_title.processing` | `""` | Tab title prefix while transcribing |
| **Console** |||
| `console.start_hidden` | `false` | Hide console after startup (whisper-key-hideable.exe only) |
| **Loading Screen** |||
| `loading_screen.enabled` | `true` | Show a startup progress window while models load (Windows only) |
| **Update** |||
| `update.mode` | `prompt` | prompt or auto |
| **Voice Commands** |||
| `voice_commands.enabled` | `true` | Enable voice command mode |

## 📁 Model Cache

faster-whisper and ONNX models are downloaded via HuggingFace to:
- **Windows:** `%USERPROFILE%\.cache\huggingface\hub\`
- **macOS:** `~/.cache/huggingface/hub/`

whisper.cpp (ggml) models live in `%LOCALAPPDATA%\whisperkey\runtimes\whisper.cpp-vulkan\models` unless
`whisper.cpp_model_dir` is set.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests target the `dev` branch; `master` only holds releases.

## Credits

Based on [Whisper Key](https://github.com/PinW/whisper-key-local) by Pin Wang (MIT). Upstream changes are merged in
periodically.

## 📦 Dependencies

**Cross-platform:**
`faster-whisper` · `ctranslate2` · `sherpa-onnx` · `onnx-asr` · `onnxruntime` · `numpy` · `sounddevice` · `soxr` · `pyperclip` · `ruamel.yaml` · `pystray` · `Pillow` · `playsound3` · `ten-vad` · `hf-xet`

**Windows:** `global-hotkeys` · `pywin32`

**macOS:** `pyobjc-framework-Quartz` · `pyobjc-framework-ApplicationServices`

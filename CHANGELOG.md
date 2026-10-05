# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## [Unreleased]

### Added
- **Voice commands from normal recordings** - with `voice_commands.match_in_dictation: true`, a normal recording whose text contains a command trigger runs that command instead of being pasted; anything else is pasted as usual. Switch it on in the tray menu (**Voice commands in normal recordings**) or with `voice_commands.match_in_dictation: true`

### Fixed
- **Qwen3-ASR int8 no longer drops long dictations** - its audio is split into chunks of at most 14 s (new per-model `max_chunk_seconds`, default 28 s), because the int8 decoder can return empty text for clips over ~15 s; an empty chunk that contains speech is logged as a warning

## [1.2.0] - 2026-10-06

### Changed
- **Model downloads use the installer window** - picking a faster-whisper model that is not downloaded yet asks first, then downloads it with progress and cancel (like ONNX and whisper.cpp models), instead of freezing while it downloads in the background
- **Transcription pauses while switching** - while a model or runtime is downloading, installing or loading, recording is off and an active recording is cancelled; the floating button and orb turn gray and faded until the new model is ready
- **Models warm up when they load** - at startup and after a switch, so the first recording is not slow and a broken GPU setup is caught while loading

### Fixed
- **Hang after a CUDA library error** - a faster-whisper model on CUDA without cuBLAS/cuDNN now fails during loading and falls back to the CPU, instead of failing the first recording and hanging on the next
- **Startup crash on an unknown model** - a model key this version does not know (for example from a newer version's settings) falls back to Tiny for the session
- **English-only models with another language** - `.en` models always transcribe in English and a warning explains that the configured language is ignored
- **Tray menu error after a settings change** - `invalid command name ...canvas` when the menu rebuilt itself while a row was highlighted
- **"Stop key activated" log spam** - only logged when a recording is actually stopped

## [1.1.0] - 2026-10-05

### Added
- **Modern tray menu** - left or right click on the tray icon opens a Windows 11 style flyout (rounded corners, light/dark theme, accent colour, switches and segmented buttons) grouped into Recording, Output and Setup, with drill-in pages for Model, Microphone, Floating button and Files and logs. Status line and mute button at the top, keyboard navigation, scrolls when taller than the screen. Toggles and options apply without closing the menu (Windows; macOS keeps the native menu with the same structure)
- **Orb skin picker** - **Floating button > Orb skin** shows a live preview of every skin; clicking one switches the orb immediately and the menu stays open to try the next (Windows)
- **Large-V3-Turbo German model** - `large-v3-turbo-german` in the model menu: a CTranslate2 conversion of primeline/whisper-large-v3-turbo-german, fine-tuned for German transcription
- **`supports_prompt: false` per model** - skips `initial_prompt` and `hotwords` for models that repeat or derail with them, such as TheStageAI's TheWhisper (faster-whisper and whisper.cpp)
- **Recording mode in the tray** - switch between toggle and push-to-talk from the tray **Recording mode** menu; the hotkeys are re-registered right away
- **Start with Windows** - tray toggle that registers Whisper Key in the per-user Run key; it uses `whisper-key-no-term.exe` when available and shows up in Task Manager's Startup tab (Windows)
- **`python -m whisper_key`** starts the app
- **Voice orb** - new `floating_widget.style: orb`: an animated, always-on-top orb that reacts to your voice level, with six looks (gold, silver, chrome, glass, aurora, black hole). Click to record, drag to move, scroll to resize, lock it and hide it in fullscreen apps from the tray. Switch between button and orb at runtime (Windows)
- **Voice commands window** - tray **Voice commands...** lists, adds, edits, reorders and deletes voice commands; records hotkey actions by pressing the keys, test-runs a command, shows which command a phrase would trigger, and saves `commands.yaml` with its comments intact, applied without a restart. New optional `enabled: false` field switches a command off (Windows)
- **Orb lock and mute buttons** - hovering the voice orb fades in two small buttons in the skin's colours: lock/unlock its position and mute/unmute the microphone. Each can be hidden from the tray (`orb_lock_button`, `orb_mute_button`) (Windows)
- **Nebula orb skin** - an energy-vortex orb with its own look for idle, recording (live waveform), processing (spiral), muted (glass sphere) and new short success ✓ / error ! states shown after a transcription or voice command; other skins flash white on success and red on error (Windows)
- **Hotkey voice commands accept F13-F24 and `page_up` / `page_down`**
- **Shortcuts window** - tray **Shortcuts...** lists every hotkey; record a primary and a secondary binding per action by pressing the keys, unset or reset them, and apply them without a restart. Duplicate bindings are blocked (Windows)

### Changed
- **Hotkeys take two bindings** - `recording_hotkey`, `command_hotkey`, `stop_key`, `auto_send_key` and `cancel_combination` are now `[primary, secondary]` lists; single-string values from older settings are still read as the primary binding. A binding already used by an earlier action is disabled at startup; an action left with no binding falls back to its default

### Fixed
- **Voice orb skins share one size** - aurora, black hole and nebula drew noticeably smaller than the other skins; every skin now looks the same size at each orb size, so switching skins no longer changes it (Windows)
- **Voice orb dragging is smooth** - the orb now follows the cursor at display rate instead of the animation frame rate (as low as 8 fps when idle) (Windows)
- **Auto-paste into apps running as administrator** - Windows silently drops the simulated input, so the app reported "Auto-pasted" while nothing arrived. The text now stays on the clipboard and a tray notification asks you to paste with Ctrl+V (Windows)

## [1.0.0] - 2026-10-05

First release of the independent fork [SesioN/whisper-key-local](https://github.com/SesioN/whisper-key-local),
based on upstream Whisper Key 0.8.3.

### Added
- **Floating record button** - always-on-top click-to-record button with microphone mute toggle (Windows)
- **Voice-activated recording** - starts on speech with 1 s pre-roll, stops after silence; sensitivity window with live speech meter
- **Loading screen** - startup progress window with elapsed time (Windows)
- **whisper.cpp engine** - Vulkan backend kept loaded via whisper-server; missing models download on demand
- **ONNX ASR engine** - Parakeet-TDT-0.6B-v3 and Qwen3-ASR-1.7B
- **Runtime and precision tray menus** - install and switch CPU / CUDA / ROCm / Vulkan runtimes on demand
- **Audio feedback toggle** in the tray
- **Release pipeline** - tagged releases are built by GitHub Actions and published with checksums
- **`whisper-key-no-term.exe`** - Windows build that never opens a console window; a native splash shows first-run setup progress and reports setup or startup failures in an error dialog, then the loading screen takes over

### Changed
- **Updates come from this fork's GitHub releases** - the in-app updater and `whisper-key self update` no longer install the upstream PyPI package
- **Auto-paste and copy to clipboard are independent tray checkboxes** - New `clipboard.copy_to_clipboard` keeps the transcription on the clipboard after auto-paste in both delivery methods (disables `paste_preserve_clipboard`); voice-command text follows the same setting. Replaces `type_also_copy_to_clipboard`, which is migrated automatically

### Fixed
- **Stale PyApp environments cleaned up** - the exe now deletes environments left behind by previous versions on startup (~400 MB each)
- **No console flashes from sound playback** - playsound3 backend probes no longer briefly open console windows on Windows

## [0.8.3] - 2026-09-27

### Fixed
- NVIDIA GPU setup now checks that cuBLAS libraries can load and finds pip-installed CUDA DLLs at startup (#68)
- AMD GPU detection now recognizes Strix Halo APUs (8040S/8050S/8060S), handles RX names without spaces, and avoids misclassifying RX 580 cards
- Added pipx-specific ROCm SDK installation instructions (#63)

## [0.8.2] - 2026-06-13

### Added
- **Text corrections** - Fix recurring misheard words with a user-defined list (`post_processing.corrections`): variants match case-insensitively and are replaced with your exact spelling — a deterministic backstop for jargon `whisper.hotwords` can't fix (#59, #60)
- **Terminal tab title status** - Tab title blinks a red dot while recording; indicator and animation per state are customizable via the new `terminal_title` config section (#61)
- **Startup ready sound** - Plays a short chime when the app finishes loading (`audio_feedback.ready_enabled`) (#62)
- **GPU failure recovery** - When GPU model loading fails, the app now offers to re-run GPU setup or fall back to CPU instead of crashing
- `post_processing.strip_trailing_period` config option to strip a trailing period from transcription output (#55)

### Changed
- **`recording_mode` now applies to the command hotkey** - push-to-talk works the same for voice commands as for transcription (#56)
- New `post_processing` config section owns output text transforms; transcribed text is now printed/logged after post-processing so the console always shows what gets pasted

## [0.8.1] - 2026-04-18

### Added
- **Push-to-talk mode** - New `hotkey.recording_mode` config: hold the recording hotkey to record, release to stop and transcribe (#45)
- **Console window management** - Two exe variants:
  - `whisper-key.exe` - runs in Windows Terminal with full color (same as before)
  - `whisper-key-hideable.exe` - conhost with hide-to-tray support (`console.start_hidden` config option) (#50)

### Fixed
- Stop key requiring two presses in toggle mode (#54)

## [0.8.0] - 2026-03-17

### Added
- **GPU onboarding** - Detects your GPU on first launch and offers one-press install of CUDA or ROCm runtime libraries. Supports NVIDIA, AMD RDNA 2+, and RDNA 1 (manual setup) (#44)
- **Auto-update** - Checks PyPI for new versions on startup with option to update in-place (#43)
- **GPU and runtime detection** - Shows GPU model and runtime status on startup (#41)
- **Overlay config** - Config updates now merge cleanly without overwriting user comments or structure (#40)
- `initial_prompt` config option to bias Whisper transcription toward expected content (#39)
- `log_transcriptions` config option (default: off) for privacy

### Changed
- **Replaced PyInstaller with pyapp** - Windows exe is now a single `whisper-key.exe` that bootstraps its own Python environment. No more zip extraction or separate AMD variant (#42)
- Updated README for new install flow

### Removed
- PyInstaller build system and related runtime hooks

## [0.7.1] - 2026-03-08

### Added
- **Transcription complete sound** - Optional audio notification when transcription finishes (`audio_feedback.transcription_complete_enabled`) (#35)
- **Custom hotwords** - Bias transcription toward specific words/phrases via `whisper.hotwords` config (#34)
- **Open model cache** shortcut in system tray menu
- CONTRIBUTING.md with PR guidelines

### Fixed
- **Portable exe crash on startup** - `commands.defaults.yaml` was not bundled in PyInstaller build, causing `[WinError 2]` crash when voice commands initialize (#37)
- Startup error handler now logs full traceback for better crash diagnostics

## [0.7.0] - 2026-02-27

### Added
- **Voice command mode** - Dedicated hotkey (`Alt+Win` / `Fn+Command`) records speech and matches against user-defined trigger phrases in `commands.yaml` (#33)
  - `run` action — execute shell commands (e.g., "open notepad" → `notepad.exe`)
  - `hotkey` action — send keyboard shortcuts (e.g., "undo" → `Ctrl+Z`)
  - `type` action — deliver pre-written text (e.g., "my email" → `user@example.com`)
  - Case-insensitive substring matching with longest-trigger-first priority
  - Auto-send support — press `Alt` to stop and send ENTER after type commands
- **Commands file shortcuts** in system tray menu for quick editing
- Terminal tab title set to "Whisper Key" on startup

### Changed
- **Hotkey config refactored** — replaced `stop_with_modifier` toggle with explicit `stop_key` and `auto_send_key` settings
- Recording hotkey is now start-only (no longer toggles)
- Cleaner console messages: one-line-per-hotkey format, indented recording output
- Renamed `documentation/` folder to `docs/`
- macOS config path moved to `~/.whisperkey`
- Updated GPU setup guide: simplified ROCm instructions, updated wheel versions

### Fixed
- ROCm runtime hook now finds pip-installed HIP SDK

## [0.6.3] - 2026-02-17

### Added
- **Direct text injection** - New `type` delivery method using native ctypes `SendInput` with `KEYEVENTF_UNICODE`, bypassing clipboard entirely (Windows only) (#21)
- **Configurable delivery method** - Choose between `paste` (clipboard + Ctrl+V) and `type` (direct key injection) via `clipboard.delivery_method`
- **Pre-paste delay** - `paste_pre_paste_delay` setting (50ms default) fixes intermittent empty paste caused by Windows Clipboard History service contention (#21)
- **Scaling auto-enter delay** - Type mode delay before Enter now scales with text length: `type_auto_enter_delay + chars/100 * type_auto_enter_delay_per_100_chars`
- **Real-time speech preview** - Experimental streaming transcription using sherpa-onnx (enable with `streaming.streaming_enabled: true`; downloads a small ~20MB model on first use)
- **Application icon** - PyInstaller exe now has a proper app icon
- **`wk` CLI alias** - Short command alias for `whisper-key`

### Fixed
- **Auto-paste empty text bug** - Clipboard race condition where Windows Clipboard History service contests clipboard between copy and paste, causing target app to receive nothing (#21)

### Changed
- Replaced pyautogui with native ctypes `SendInput` for keyboard simulation on Windows (smaller dependency footprint, atomic key injection)
- Restructured clipboard config: `paste_*` prefix for paste-mode settings, `type_*` for type-mode settings, `macos_*` for macOS-only
- Delivery method validation moved into platform keyboard modules
- Default `paste_clipboard_restore_delay` set to 0.5s

### Removed
- **pyautogui** dependency — replaced by native ctypes

## [0.6.2] - 2026-02-09

### Added
- **AMD GPU portable exe** - Separate ROCm build variant for AMD RX 5000+ GPUs
- **CPU/GPU mode display** - Shows device mode and compute type on startup
- **Distil-whisper models** - Added distil-medium.en and distil-small.en to default config

### Fixed
- **Model load crash** - Bundled correct MSVCP140.dll instead of PyInstaller's incompatible version (#22)
- **Tray icon path** - Fixed icon resolution for PyInstaller builds using `resolve_asset_path()`

### Changed
- Updated GPU setup guide with pip, pipx, and portable exe instructions for AMD GPUs
- Updated README with AMD GPU download variant

## [0.6.1] - 2026-02-04

### Added
- **macOS support** - Full platform abstraction layer with native integration for hotkeys (NSEvent), keyboard simulation (Quartz CGEvent), and system tray (#23)
  - Fn key modifier support for hotkeys
  - Accessibility permission prompt for auto-paste
  - Platform-specific default hotkeys: fn+ctrl (record), shift (cancel), cmd+v (paste)
- `--test` flag for running a separate test instance alongside the main app
- CUDA setup instructions in config file comments (cuDNN no longer required)

### Changed
- Reduced console verbosity by removing config update messages
- ten-vad is now a standard PyPI dependency (no separate install step needed)
- Updated README with macOS installation and usage instructions

### Fixed
- UTF-8 stdout encoding for special characters on Windows
- Audio feedback sounds going silent after idle periods on Windows (switched to winmm backend)
- PyPI package missing platform-specific tray icons
- PyInstaller build: ten_vad path and Windows platform assets

### Dependencies
- **ten-vad**: Now `>=1.0.6` (was git-only)
- **pyobjc-framework-Quartz**: Added for macOS (keyboard simulation)
- **pyobjc-framework-ApplicationServices**: Added for macOS (Accessibility permissions)

## [0.5.3] - 2026-01-19

### Fixed
- PyInstaller crash with ctranslate2 4.6.3 due to bundled MSVCP140.dll version mismatch

### Changed
- Replaced scipy with soxr for audio resampling (smaller bundle size)

### Dependencies
- **scipy**: Removed - no longer required
- **soxr**: Added `>=0.5.0` for high-quality audio resampling

## [0.5.2] - 2026-01-16

### Fixed
- Single-key hotkeys (e.g., F13) causing "hotkey already registered" error when `stop_with_modifier_enabled` was true (#14)

## [0.5.1] - 2026-01-16

### Fixed
- User settings file was showing "DO NOT EDIT THIS FILE" header from the defaults template

### Added
- Configurable paste hotkey setting (`clipboard.paste_hotkey`) - workaround for Claude Code changing CTRL+V to CTRL+SHIFT+V (possibly a bug)
- Console messages when opening log file and settings from system tray

## [0.5.0] - 2026-01-15

### Added
- **Custom local model support** - Load any Whisper model from a local folder or HuggingFace path (#10)
- **New models**: large-v3-turbo and distil-large-v3.5
- **View Log** shortcut in system tray menu (#9)
- **Advanced Settings** shortcut in system tray menu

### Changed
- Config key renamed from `whisper.model_size` to `whisper.model`
- Model menu in system tray now built dynamically with grouped separators
- Updated model download sizes to reflect actual faster-whisper downloads

### Fixed
- Cache detection for non-Systran models (e.g., large-v3-turbo showed "Downloading..." when already cached)
- Config section headers duplicating on save

### Dependencies
- **faster-whisper**: `>=1.1.1` → `>=1.2.1`
  Required for distil-large-v3.5 support
- **ctranslate2**: Added explicit `>=4.6.3` requirement
  May fix GPU crashes on systems without cuDNN installed (#15)

## [0.4.0] - 2026-01-14

### Added
- Audio source selection from system tray menu (WASAPI devices)
- WASAPI loopback support with bundled custom PortAudio DLL for recording system audio
- Version display in startup message

### Fixed
- WASAPI devices that don't support 16kHz now work via automatic resampling
- WASAPI stream reopen race condition with OS-level cleanup delay
- User feedback when audio device switch fails

## [0.3.0] - 2025-08-19

### Added
- Complete PyPI package distribution support with robust asset resolution
- Audio recording duration display
- Cancel recording hotkey with distinct sound feedback

### Changed
- Unified version tracking from pyproject.toml
- Centralized user data directory with proper app.log location handling
- Enhanced release script to include git push and use version-specific changelog notes
- Updated README with new install instructions

### Fixed
- Various typos and documentation inconsistencies

## [0.2.0] - 2025-08-15

### Added
- Max recording duration with callback-based duration limiting for audio capture
- Intelligent hotkey conflict detection with automatic resolution
- Global exception handling with stderr redirection to app.log
- VAD configuration reorganization into dedicated Voice Activity Detection section
- Instance Manager component (renamed from single_instance.py for consistency)

### Changed
- Massive code reduction: 50% reduction across 11 core files (4,200 → 2,087 lines)
- Complete removal of redundant docstrings and beginner-friendly comments
- Simplified component interfaces and eliminated circular dependencies
- Enhanced build script with improved Start Menu compatibility and asset path resolution
- Updated automation workflow to focus on real issues over defensive programming
- Exception handling standardization across all components
- Import reorganization following PEP 8 standards
- Magic number extraction to named constants
- Component interface simplification with better separation of concerns

### Fixed
- Critical race conditions in transcription pipeline and pending model changes
- Bare except clause that could mask critical exceptions
- Inconsistent OptionalComponent usage in SystemTray type annotations

### Removed
- Entire test suite to focus on shipping over ceremony
- Defensive programming patterns and unnecessary validation
- Windows API clipboard fallback complexity
- Complex model loading progress tracking
- Redundant configuration options and unused settings
- Dead code and unused functions throughout codebase

## [0.1.3] - 2025-08-11

### Added
- Comprehensive CHANGELOG.md for project releases
- Version bump command for Claude AI assistant

### Changed
- Updated build instructions and removed redundant builder.py

### Fixed
- Single instance detector exit error in built executable

## [0.1.2] - 2025-08-11

### Added
- Single-instance detection with mutex to prevent multiple app instances
- TEN VAD (Voice Activity Detection) pre-check system with advanced post-processing
- Audio feedback component for recording events
- PyInstaller packaging system for Windows executable distribution
- Auto-enter hotkey functionality with configurable modifiers
- Stop-with-modifier hotkey functionality
- Alternative keycode checker tool for hotkey configuration
- Context manager for consistent error handling across components

### Changed
- Default Whisper model changed from `tiny` to `base` for better accuracy
- Default hotkeys updated to CTRL+WIN+SPACE (record) and CTRL+WIN+SHIFT+SPACE (stop)
- Renamed `auto_enter_delay` to `key_simulation_delay` for clarity
- Simplified clipboard copy operation for better performance
- Default console logging level set to warning for regular users
- Disabled UPX compression in PyInstaller to reduce antivirus false positives
- Improved system tray model selection to show actual model names
- Streamlined user feedback messages around clipboard actions and startup

### Fixed
- CTRL+C unresponsiveness by adding proper hotkey listener cleanup during shutdown
- Windows key support in hotkey detection
- Auto-enter hotkey now respects auto-paste setting
- Config validation now persists properly to user settings file
- YAML structure corruption by using clean config template
- Unicode logging errors on Windows
- Executable Start Menu launch bug with working directory
- Module imports after directory refactor

### Removed
- `suppress_warnings` config option and related code
- Redundant system tray print statements
- Duplicate "Ready to paste" message

## [0.1.1] - 2025-07-15

### Added
- Complete working whisper speech-to-text application
- Comprehensive configuration system with YAML support
- Interactive key helper utility for hotkey configuration
- Auto-paste feature with Windows API integration
- System tray icon functionality with visual recording status
- User settings system with system tray controls
- Clipboard preservation feature
- Model selection submenu in system tray
- English-specific model options
- Async model loading with responsive system tray
- Tool to clear application log file
- Tool to clear model cache

### Changed
- Refactored project structure and updated documentation
- Renamed main.py to whisper-key.py for better clarity
- Renamed log file from whisper_app.log to app.log
- Updated to use faster-whisper framework
- Improved model download messaging with cache detection

### Fixed
- Module imports and directory structure issues
- System tray menu organization and cleanup

## [0.1.0] - 2025-06-01

### Added
- Initial project setup
- Core speech-to-text functionality
- Basic hotkey detection
- Audio recording capabilities
- Clipboard integration

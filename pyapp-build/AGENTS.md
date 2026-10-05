Builds `whisper-key.exe` using [pyapp](https://github.com/ofek/pyapp) (Rust wrapper that bootstraps Python + pip install at first launch).

## Prerequisites

- Windows with Rust toolchain (`cargo`)
- [pyapp source](https://github.com/ofek/pyapp/releases/latest) extracted locally

## Setup

Copy `build-config.example.json` to `build-config.json` and set paths:
- `pyapp_source_path`: where you extracted pyapp source
- `dist_path`: output directory for built exe

## Build

Use `$build-pyapp-exe` or run directly:

```powershell
powershell.exe -ExecutionPolicy Bypass -File pyapp-build/build-pyapp.ps1
```

`-Clean` flag forces full Rust rebuild.

`-FromSource` embeds a wheel built from the current commit (version `<version>+g<commit>`) instead of installing
`whisper-key-local` from PyPI. Use it for builds of unreleased branches; uncommitted changes, untracked files and `export-ignore` paths
are not included (the source is taken from `git archive HEAD`). Building the wheel needs `git`, `python` and network
access for the build dependencies. Each commit installs into its own pyapp directory.

`whisper-key self update` is patched out: it would install upstream's PyPI package. The app updates itself from
the fork's GitHub releases (`update_checker.py`).

Produces three executables: `whisper-key.exe` (console), `whisper-key-hideable.exe` (GUI subsystem, for start_hidden/minimize-to-tray) and `whisper-key-no-term.exe` (GUI subsystem, never opens a console; only the loading screen, tray and widget are visible).

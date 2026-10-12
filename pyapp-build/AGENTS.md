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

`-Release` (with `-FromSource`) gives the embedded wheel the plain `pyproject.toml` version instead of
`<version>+g<commit>`. The GitHub release workflow uses it for tagged builds.

Produces two executables: `whisper-key.exe` (GUI subsystem, never opens a console; only the loading screen, tray and widget are visible) and `whisper-key-console.exe` (console, shows the log).

## Integrity and diagnostics

Fetch fresh sources with `fetch-pyapp-source.ps1 -Destination <empty-directory>`;
`-ArchivePath <source.tar.gz>` verifies a cached archive without network access.
`pyapp-source.json` pins PyApp 0.29.0 and the SHA256 of the locally retained release archive.
The upstream asset digest could not be independently retrieved in the restricted environment;
verify it against the upstream release asset before the first release using this pin.

`Cargo.lock` records the existing patched PyApp dependency graph (winresource, windows-sys,
jpeg-decoder). Changes to these patches must update/review this lockfile; builds use `--locked`.
Install Python build tools (`build`, `setuptools`, `wheel`) before building. Wheel creation
uses `--no-build-isolation` so published `BUILD-PYTHON-DEPENDENCIES.txt` records the actual
build environment. It does not lock first-launch application dependencies.

`-Diagnostics` adds `whisper-key-diagnostic-no-splash.exe`, sharing the wheel and flags
with the GUI build while removing only the launcher's animated splash. Use separate output
and PyApp source directories. Manual release workflows can request this artifact; tag
releases build only the normal GUI and console variants. `BUILD-INFO.json` and
`BUILD-CARGO.lock` preserve toolchain/source metadata. Diagnostics and VM acceptance:
[windows-antivirus.md](../docs/windows-antivirus.md).

# Windows antivirus reports

Whisper Key uses a Rust PyApp launcher. On first launch it downloads Python from
`astral-sh/python-build-standalone` on GitHub, extracts its embedded application wheel,
and installs Python dependencies. GPU runtimes and models download separately when selected.
The app uses global hotkeys, microphone capture and Windows input APIs to deliver text.

## Collect the actual detection

Record the release version, original download URL, filename, SHA256, detection time,
Microsoft product, threat name and affected resource path. Obtain the threat name from
Windows Security > Virus & threat protection > Protection history. A warning about an
unknown publisher or an uncommon download is a SmartScreen reputation warning; a
named threat/quarantine is a Defender Antivirus finding. Smart App Control is separate.

On the affected Windows machine, these commands collect evidence without changing protection:

```powershell
Get-FileHash -LiteralPath .\whisper-key.exe -Algorithm SHA256
Get-AuthenticodeSignature -LiteralPath .\whisper-key.exe | Select-Object Status, StatusMessage, SignerCertificate
Get-MpComputerStatus | Select-Object AMEngineVersion, AntivirusSignatureVersion, AntivirusSignatureLastUpdated, RealTimeProtectionEnabled
Get-MpThreatDetection | Select-Object ThreatID, InitialDetectionTime, Resources, ActionSuccess
```

If the affected resource is a downloaded Python executable, DLL, runtime or package,
include that exact file/hash as well as the launcher. A scan of the launcher alone
does not establish which later download was detected. Keep personal paths and user
settings out of public reports.

## Release verification and limitations

Updates select one wheel matching this project's release version and verify its SHA256
against that release's `SHA256SUMS.txt` before pip installs the local file. Missing,
ambiguous or malformed metadata, interrupted downloads and mismatches fail before
installation. The release manifest and wheel share GitHub as their trust source:
this is integrity checking, not independent publisher authentication. pip can still
download dependencies; those downloads are not covered by the application wheel's hash.
A pip installation failure can partially change the environment; verification failures
leave the installed application alone.

Current releases have no configured Authenticode signing. Do not describe a checksum
match or a clean sandbox verdict as proof that every dependency or every execution path
has been vetted. The targeted changes do not guarantee zero antivirus detections.

## Prepare a developer submission

Use [Microsoft Security Intelligence](https://www.microsoft.com/en-us/wdsi/filesubmission)
as a Software developer, choose the actual Microsoft product and report an incorrect
detection. Microsoft's [developer FAQ](https://learn.microsoft.com/en-us/defender-xdr/developer-faq)
describes disputing a determination after the submission completes.

Include the exact detected binary, its hash and signature status, source commit/tag,
release URL, detection details, reproduction steps and any behavior report. Submit the
existing affected bytes; rebuilding or signing changes the hash and creates another sample.
Prepare equivalent submissions only for vendors whose detections are confirmed.
Never send user settings, recordings or logs containing private transcriptions.

Suggested description:

> Whisper Key is an open-source local speech-to-text app. Its Rust PyApp launcher
> installs Python and dependencies on first launch; models and GPU runtimes are
> downloaded when selected. Please review the attached exact binary for an incorrect
> detection. Source: https://github.com/SesioN/whisper-key-local.
> Release/tag/commit: [fill in]. SHA256: [fill in]. Product/threat name: [fill in].
> Detection stage and affected path: [fill in]. Reproduction: [fill in].

## Controlled packaging comparison

Use `build-pyapp.ps1 -FromSource -Diagnostics` in a dedicated source/output directory.
It produces normal GUI and console launchers plus `whisper-key-diagnostic-no-splash.exe`.
The diagnostic variant removes only the Rust launcher's splash implementation/image;
the Python application's splash and UI remain. All variants share one embedded wheel,
PyApp source, lockfile and toolchain. Diagnostic builds are manual workflow artifacts,
not automatically published releases.

For each file, record SHA256 and signature status. Run
`python tools/security/inspect_windows_artifact.py <exe> [<exe> ...] --output <report.json>`
to compare file entropy, PE section names/sizes and directory flags. An embedded archive
or compressed image can resemble packed data; the heuristic label does not prove packing.
Do not remove sections or rename them simply to hide a detection.

Test in fresh snapshots of a Windows 11 VM with current Defender protection:

| Stage | Record |
|---|---|
| Browser download and static scan | Product, threat name, exact hash, affected path |
| First launch | Python/dependency downloads and whether the app reaches ready state |
| Second launch | Behavior after bootstrap is already installed |
| Update | Verified wheel install and restart, or safe verification failure |
| Runtime/model install | Any separately detected downloaded file/hash |
| Recording/delivery | Hotkeys, microphone, clipboard/typing, tray and voice commands |
| Autostart and GUI handover | Opt-in registration, splash handover, error visibility |

Take the same baseline snapshot for each variant. Keep scanning separate from launch
and retain engine/signature versions with results. The release workflow reuses the embedded application wheel as the published wheel.
Scan and publish the same final artifact bytes. Check again after vendor corrections. Do not claim a variant resolves
false positives until these tests confirm it.

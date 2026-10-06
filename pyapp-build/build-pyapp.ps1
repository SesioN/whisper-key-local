param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$AppName = "whisper-key",
    [switch]$Clean,
    [switch]$FromSource,
    [switch]$Release
)

function Get-ResolvedPath {
    param($Path, $BaseDir)
    if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $BaseDir $Path }
}

function Get-ProjectVersion {
    param($ProjectRoot)
    $PyProjectFile = Join-Path $ProjectRoot "pyproject.toml"
    foreach ($Line in (Get-Content $PyProjectFile)) {
        if ($Line.StartsWith("version")) {
            return $Line.Split("=")[1].Trim().Trim('"')
        }
    }
    Write-Host "Error: Could not find version in pyproject.toml" -ForegroundColor Red
    exit 1
}

function Build-SourceWheel {
    param($ProjectRoot, $AppVersion, $DistPath)

    foreach ($Tool in "git", "python") {
        if (-not (Get-Command $Tool -ErrorAction SilentlyContinue)) {
            Write-Host "Error: -FromSource needs $Tool on PATH" -ForegroundColor Red
            exit 1
        }
    }

    $Commit = git -C $ProjectRoot rev-parse --short HEAD
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: -FromSource needs a git checkout" -ForegroundColor Red
        exit 1
    }
    $Commit = $Commit.Trim()
    if (git -C $ProjectRoot status --porcelain) {
        Write-Host "Warning: uncommitted changes are not included (building commit $Commit)" -ForegroundColor Yellow
    }

    $WheelVersion = if ($Release) { $AppVersion } else { "$AppVersion+g$Commit" }
    $StagingDir = Join-Path ([System.IO.Path]::GetTempPath()) "whisper-key-wheel-$([guid]::NewGuid())"
    $WheelDir = Join-Path $DistPath "wheel"
    if (Test-Path $WheelDir) { Remove-Item -Recurse -Force $WheelDir }
    New-Item -ItemType Directory -Path $StagingDir, $WheelDir -Force | Out-Null

    try {
        $Archive = Join-Path $StagingDir "source.zip"
        git -C $ProjectRoot archive --format=zip --output=$Archive HEAD
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: git archive failed" -ForegroundColor Red
            exit 1
        }
        Expand-Archive -Path $Archive -DestinationPath $StagingDir -ErrorAction Stop
        Remove-Item $Archive

        $PyProjectFile = Join-Path $StagingDir "pyproject.toml"
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        $PyProject = [System.IO.File]::ReadAllText($PyProjectFile, $Utf8NoBom)
        $VersionPattern = [regex]'(?m)^version\s*=\s*"[^"]+"'
        $PatchedPyProject = $VersionPattern.Replace($PyProject, "version = `"$WheelVersion`"", 1)
        if ($PatchedPyProject -eq $PyProject -and -not $Release) {
            Write-Host "Error: Could not set wheel version in pyproject.toml" -ForegroundColor Red
            exit 1
        }
        [System.IO.File]::WriteAllText($PyProjectFile, $PatchedPyProject, $Utf8NoBom)

        Write-Host "Building wheel $WheelVersion from commit $Commit..." -ForegroundColor Yellow
        python -m pip wheel $StagingDir --no-deps --wheel-dir $WheelDir --quiet | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Wheel build failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Remove-Item -Recurse -Force $StagingDir -ErrorAction SilentlyContinue
    }

    $Wheel = Get-ChildItem $WheelDir -Filter "whisper_key_local-$WheelVersion-*.whl" | Select-Object -First 1
    if (-not $Wheel) {
        Write-Host "Error: No wheel for $WheelVersion found in $WheelDir" -ForegroundColor Red
        exit 1
    }
    return $Wheel.FullName
}

function Patch-IconSupport {
    param($PyAppSourcePath, $IconPath)

    $IconDest = Join-Path $PyAppSourcePath "icon.ico"
    Copy-Item $IconPath $IconDest -Force

    $CargoToml = Join-Path $PyAppSourcePath "Cargo.toml"
    $CargoContent = Get-Content $CargoToml -Raw
    if ($CargoContent -notmatch "winresource") {
        $CargoContent = $CargoContent -replace '\[build-dependencies\]', "[build-dependencies]`nwinresource = `"0.1`""
        Set-Content $CargoToml $CargoContent -NoNewline
        Write-Host "Patched Cargo.toml with winresource dependency" -ForegroundColor Gray
    }

    $BuildRs = Join-Path $PyAppSourcePath "build.rs"
    $BuildContent = Get-Content $BuildRs -Raw
    if ($BuildContent -notmatch "winresource") {
        $IconPatch = @'

    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("icon.ico");
        res.compile().expect("Failed to compile Windows resources");
    }
'@
        $BuildContent = $BuildContent -replace '(fn main\(\) \{)', "`$1$IconPatch"
        Set-Content $BuildRs $BuildContent -NoNewline
        Write-Host "Patched build.rs with icon embedding" -ForegroundColor Gray
    }
}

function Set-PatchedText {
    param($Path, $Find, $Replace, $Marker)
    $Find = $Find -replace "`r`n", "`n"
    $Replace = $Replace -replace "`r`n", "`n"
    $Marker = $Marker -replace "`r`n", "`n"
    $Content = [System.IO.File]::ReadAllText($Path) -replace "`r`n", "`n"
    if ($Content.Contains($Marker)) { return }
    if (-not $Content.Contains($Find)) {
        Write-Host "Error: patch anchor not found in $Path" -ForegroundColor Red
        exit 1
    }
    $Index = $Content.IndexOf($Find)
    $Content = $Content.Substring(0, $Index) + $Replace + $Content.Substring($Index + $Find.Length)
    [System.IO.File]::WriteAllText($Path, $Content)
}

function Patch-SelfUpdate {
    param($PyAppSourcePath)
    Set-PatchedText (Join-Path $PyAppSourcePath "src\commands\self_cmd\update.rs") @'
    pub fn exec(self) -> Result<()> {
'@ @'
    pub fn exec(self) -> Result<()> {
        println!("Whisper Key installs updates from https://github.com/SesioN/whisper-key-local/releases when it starts");
        exit(1);
'@ 'Whisper Key installs updates from'
}

function Patch-NoTerminalSupport {
    param($PyAppSourcePath)

    Copy-Item (Join-Path $PSScriptRoot "pyapp-patches\splash.rs") (Join-Path $PyAppSourcePath "src\splash.rs") -Force
    Copy-Item (Join-Path $ProjectRoot "src\whisper_key\platform\windows\assets\splash_frames.jpg") (Join-Path $PyAppSourcePath "src\splash_frames.jpg") -Force

    Set-PatchedText (Join-Path $PyAppSourcePath "Cargo.toml") '[build-dependencies]' @'
[target.'cfg(windows)'.dependencies]
windows-sys = { version = "0.59", features = ["Win32_Foundation", "Win32_Graphics_Gdi", "Win32_System_Diagnostics_ToolHelp", "Win32_System_LibraryLoader", "Win32_UI_WindowsAndMessaging"] }

[build-dependencies]
'@ 'windows-sys = { version = "0.59"'
    Set-PatchedText (Join-Path $PyAppSourcePath "Cargo.toml") '"Win32_System_LibraryLoader", "Win32_UI_WindowsAndMessaging"] }' '"Win32_System_Diagnostics_ToolHelp", "Win32_System_LibraryLoader", "Win32_UI_WindowsAndMessaging"] }' 'Win32_System_Diagnostics_ToolHelp'
    Set-PatchedText (Join-Path $PyAppSourcePath "Cargo.toml") 'windows-sys = { version = "0.59", features = [' 'windows-sys = { version = "0.59", features = ["Win32_Graphics_Dwm", ' 'Win32_Graphics_Dwm'
    Set-PatchedText (Join-Path $PyAppSourcePath "Cargo.toml") "[target.'cfg(windows)'.dependencies]`n" "[target.'cfg(windows)'.dependencies]`njpeg-decoder = { version = `"0.3`", default-features = false }`n" 'jpeg-decoder'

    Set-PatchedText (Join-Path $PyAppSourcePath "build.rs") 'fn main() {' @'
fn main() {
    println!("cargo:rustc-check-cfg=cfg(pyapp_windows_subsystem)");
    println!("cargo:rerun-if-env-changed=PYAPP_WINDOWS_SUBSYSTEM");
    if std::env::var("PYAPP_WINDOWS_SUBSYSTEM").is_ok_and(|value| value == "1" || value == "true") {
        println!("cargo:rustc-cfg=pyapp_windows_subsystem");
    }
'@ 'pyapp_windows_subsystem'

    $MainRs = Join-Path $PyAppSourcePath "src\main.rs"
    Set-PatchedText $MainRs 'mod app;' @'
#![cfg_attr(all(windows, pyapp_windows_subsystem), windows_subsystem = "windows")]

mod app;
mod splash;
'@ 'mod splash;'
    Set-PatchedText $MainRs 'fn main() -> Result<()> {' @'
fn main() -> Result<()> {
    let result = run();
    if cfg!(pyapp_windows_subsystem) {
        if let Err(error) = &result {
            splash::show_error(&format!("{error:#}"));
            std::process::exit(1);
        }
    }
    result
}

fn run() -> Result<()> {
'@ 'fn run() -> Result<()> {'

    $TerminalRs = Join-Path $PyAppSourcePath "src\terminal.rs"
    Set-PatchedText $TerminalRs 'let pb = ProgressBar::new(size);' @'
crate::splash::set_status(&message);
    let pb = ProgressBar::new(size);
'@ 'crate::splash::set_status(&message);
    let pb'
    Set-PatchedText $TerminalRs 'let s = ProgressBar::new(0);' @'
crate::splash::set_status(&message);
    let s = ProgressBar::new(0);
'@ 'crate::splash::set_status(&message);
    let s'

    $ProcessRs = Join-Path $PyAppSourcePath "src\process.rs"
    Set-PatchedText $ProcessRs 'use crate::{app, terminal};' @'
use crate::{app, terminal};

#[cfg(windows)]
fn hide_console_window(command: &mut Command) {
    if cfg!(pyapp_windows_subsystem) {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

#[cfg(not(windows))]
fn hide_console_window(_command: &mut Command) {}
'@ 'fn hide_console_window'
    Set-PatchedText $ProcessRs @'
    command.stderr(writer_stderr);

    let mut child = command.spawn()?;
'@ @'
    command.stderr(writer_stderr);
    hide_console_window(&mut command);

    let mut child = command.spawn()?;
'@ 'hide_console_window(&mut command);

    let mut child = command.spawn()?;
    drop'
    Set-PatchedText $ProcessRs @'
fn exec_gui(mut command: Command) -> Result<()> {
    let mut child = command.spawn()?;
'@ @'
fn exec_gui(mut command: Command) -> Result<()> {
    hide_console_window(&mut command);
    let mut child = command.spawn()?;
    crate::splash::hand_over(child.id());
'@ 'crate::splash::hand_over'
    Set-PatchedText $ProcessRs @'
    hide_console_window(&mut command);
    let mut child = command.spawn()?;
    crate::splash::hand_over
'@ @'
    hide_console_window(&mut command);
    crate::splash::prepare_child(&mut command);
    let mut child = command.spawn()?;
    crate::splash::hand_over
'@ 'crate::splash::prepare_child'
}

$AppVersion = Get-ProjectVersion $ProjectRoot
Write-Host "Version: $AppVersion" -ForegroundColor Cyan

$ConfigFile = Join-Path $PSScriptRoot "build-config.json"
if (-not (Test-Path $ConfigFile)) {
    Write-Host "Error: Build config not found at $ConfigFile" -ForegroundColor Red
    Write-Host "Copy build-config.example.json to build-config.json and set your paths" -ForegroundColor Yellow
    exit 1
}

$Config = Get-Content $ConfigFile | ConvertFrom-Json
$PyAppSourcePath = Get-ResolvedPath $Config.pyapp_source_path $ProjectRoot
$DistPath = Get-ResolvedPath $Config.dist_path $ProjectRoot

if (-not (Test-Path (Join-Path $PyAppSourcePath "Cargo.toml"))) {
    Write-Host "Error: No Cargo.toml found in $PyAppSourcePath" -ForegroundColor Red
    Write-Host "Download from: https://github.com/ofek/pyapp/releases/latest/download/source.tar.gz" -ForegroundColor Yellow
    exit 1
}

$IconPath = Join-Path $ProjectRoot "src\whisper_key\platform\windows\assets\whisperkey-icon.ico"
if (Test-Path $IconPath) {
    Write-Host "Patching icon support..." -ForegroundColor Yellow
    Patch-IconSupport $PyAppSourcePath $IconPath
} else {
    Write-Host "Warning: Icon not found at $IconPath, building without icon" -ForegroundColor Yellow
}

Write-Host "Patching no-terminal support..." -ForegroundColor Yellow
Patch-NoTerminalSupport $PyAppSourcePath

Write-Host "Disabling pyapp self update..." -ForegroundColor Yellow
Patch-SelfUpdate $PyAppSourcePath

Write-Host "Starting pyapp build for $AppName v$AppVersion..." -ForegroundColor Green
Write-Host "PyApp source: $PyAppSourcePath" -ForegroundColor Gray
Write-Host "Distribution: $DistPath" -ForegroundColor Gray

$PyAppVars = "PYAPP_PROJECT_NAME", "PYAPP_PROJECT_VERSION", "PYAPP_PROJECT_PATH", "PYAPP_PYTHON_VERSION", "PYAPP_EXEC_CODE", "PYAPP_SELF_COMMAND", "PYAPP_PASS_LOCATION", "PYAPP_IS_GUI", "PYAPP_WINDOWS_SUBSYSTEM"
$SavedEnv = @{}
foreach ($Var in $PyAppVars) { $SavedEnv[$Var] = [Environment]::GetEnvironmentVariable($Var) }
$PushedLocation = $false

try {
    if ($FromSource) {
        $env:PYAPP_PROJECT_PATH = Build-SourceWheel $ProjectRoot $AppVersion $DistPath
        Write-Host "Embedding $($env:PYAPP_PROJECT_PATH)" -ForegroundColor Gray
        Remove-Item Env:\PYAPP_PROJECT_NAME -ErrorAction SilentlyContinue
        Remove-Item Env:\PYAPP_PROJECT_VERSION -ErrorAction SilentlyContinue
    } else {
        Remove-Item Env:\PYAPP_PROJECT_PATH -ErrorAction SilentlyContinue
        $env:PYAPP_PROJECT_NAME = "whisper-key-local"
        $env:PYAPP_PROJECT_VERSION = $AppVersion
    }
    $env:PYAPP_PYTHON_VERSION = "3.12"
    $ExecCode = 'from whisper_key.main import main; main()'
    $NoTerminalExecCode = 'import os; os.environ["WHISPER_KEY_NO_TERMINAL"] = "1"; from whisper_key.main import main; main()'
    $env:PYAPP_SELF_COMMAND = "self"
    $env:PYAPP_PASS_LOCATION = "true"

    if ($Clean) {
        $TargetDir = Join-Path $PyAppSourcePath "target"
        if (Test-Path $TargetDir) {
            Write-Host "Cleaning previous Rust build..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $TargetDir
        }
    }

    if (-not (Test-Path $DistPath)) {
        New-Item -ItemType Directory -Path $DistPath -Force | Out-Null
    }

    $Builds = @(
        @{ Name = "$AppName";         IsGui = $true;  NoTerminal = $true;  Label = "no terminal (GUI subsystem)"; ExecCode = $NoTerminalExecCode },
        @{ Name = "$AppName-console"; IsGui = $false; NoTerminal = $false; Label = "console";                     ExecCode = $ExecCode }
    )

    Push-Location $PyAppSourcePath
    $PushedLocation = $true

    foreach ($Build in $Builds) {
        Write-Host "`nBuilding $($Build.Label): $($Build.Name).exe..." -ForegroundColor Yellow

        $env:PYAPP_EXEC_CODE = $Build.ExecCode
        if ($Build.IsGui) { $env:PYAPP_IS_GUI = "true" }
        else { Remove-Item Env:\PYAPP_IS_GUI -ErrorAction SilentlyContinue }
        if ($Build.NoTerminal) { $env:PYAPP_WINDOWS_SUBSYSTEM = "1" }
        else { Remove-Item Env:\PYAPP_WINDOWS_SUBSYSTEM -ErrorAction SilentlyContinue }

        (Get-Item (Join-Path $PyAppSourcePath "build.rs")).LastWriteTime = Get-Date

        cargo build --release
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Build failed for $($Build.Name)!" -ForegroundColor Red
            exit 1
        }

        $SourceExe = Join-Path $PyAppSourcePath "target\release\pyapp.exe"
        $DestExe = Join-Path $DistPath "$($Build.Name).exe"
        Copy-Item $SourceExe $DestExe -Force

        $ExeSize = (Get-Item $DestExe).Length / 1MB
        Write-Host ("  -> $DestExe ({0:N2} MB)" -f $ExeSize) -ForegroundColor Green
    }

    Write-Host "`nBuild complete!" -ForegroundColor Green
} finally {
    if ($PushedLocation) { Pop-Location }
    foreach ($Var in $PyAppVars) { [Environment]::SetEnvironmentVariable($Var, $SavedEnv[$Var]) }
}

param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$AppName = "whisper-key",
    [switch]$Clean,
    [switch]$FromSource
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

    $WheelVersion = "$AppVersion+g$Commit"
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
        if ($PatchedPyProject -eq $PyProject) {
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

    $Wheel = Get-ChildItem $WheelDir -Filter "*+g$Commit-*.whl" | Select-Object -First 1
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

Write-Host "Starting pyapp build for $AppName v$AppVersion..." -ForegroundColor Green
Write-Host "PyApp source: $PyAppSourcePath" -ForegroundColor Gray
Write-Host "Distribution: $DistPath" -ForegroundColor Gray

$PyAppVars = "PYAPP_PROJECT_NAME", "PYAPP_PROJECT_VERSION", "PYAPP_PROJECT_PATH", "PYAPP_PYTHON_VERSION", "PYAPP_EXEC_CODE", "PYAPP_SELF_COMMAND", "PYAPP_PASS_LOCATION", "PYAPP_IS_GUI"
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
    $env:PYAPP_EXEC_CODE = 'from whisper_key.main import main; main()'
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
        @{ Name = "$AppName";      IsGui = $false; Label = "console" },
        @{ Name = "$AppName-hideable"; IsGui = $true;  Label = "hideable (GUI subsystem)" }
    )

    Push-Location $PyAppSourcePath
    $PushedLocation = $true

    foreach ($Build in $Builds) {
        Write-Host "`nBuilding $($Build.Label): $($Build.Name).exe..." -ForegroundColor Yellow

        if ($Build.IsGui) { $env:PYAPP_IS_GUI = "true" }
        else { Remove-Item Env:\PYAPP_IS_GUI -ErrorAction SilentlyContinue }

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

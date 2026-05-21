param(
    [string[]]$Versions = @("13.0", "13.1", "13.2", "14.0", "14.1", "15.0", "15.1", "15.2", "16.0", "17.0"),
    [switch]$DeployToNuke,
    [string]$NukeBinRoot = "$env:USERPROFILE\.nuke\TBlur\bin"
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-VsDevCmdPath {
    $candidates = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "VsDevCmd.bat introuvable. Installe Visual Studio Build Tools 2022 avec le workload C++."
}

function Invoke-InVsDevShell {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    $vsDevCmd = Get-VsDevCmdPath
    $cmdLine = "call `"$vsDevCmd`" -arch=amd64 -host_arch=amd64 -no_logo >nul && $Command"
    & cmd.exe /d /s /c $cmdLine
    if ($LASTEXITCODE -ne 0) {
        throw "Commande echouee dans le shell Visual Studio: $Command"
    }
}

Push-Location $scriptRoot
try {
    foreach ($version in $Versions) {
        Write-Host ""
        Write-Host "=== Building Nuke $version (windows) ===" -ForegroundColor Cyan
        Invoke-InVsDevShell "cargo xtask --compile --nuke-versions $version --target-platform windows --output-to-package --limit-threads --cuda-backend"
    }

    if ($DeployToNuke) {
        $pluginSrcRoot = Join-Path $scriptRoot "tblur_plugins\tblur_plugin"
        $srcRoot = Join-Path $pluginSrcRoot "bin"
        $nukePluginRoot = Split-Path -Parent $NukeBinRoot

        if (-not (Test-Path -LiteralPath $pluginSrcRoot)) {
            throw "Package source introuvable: $pluginSrcRoot"
        }

        New-Item -ItemType Directory -Path $nukePluginRoot -Force | Out-Null
        New-Item -ItemType Directory -Path $NukeBinRoot -Force | Out-Null

        # Sync plugin python/resources files (everything except bin).
        Get-ChildItem -LiteralPath $pluginSrcRoot -Force |
            Where-Object { $_.Name -ne "bin" } |
            ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $nukePluginRoot -Recurse -Force
            }

        # Sync selected compiled binaries per requested version.
        foreach ($version in $Versions) {
            $src = Join-Path $srcRoot $version
            if (Test-Path $src) {
                try {
                    Copy-Item -LiteralPath $src -Destination $NukeBinRoot -Recurse -Force
                }
                catch {
                    Write-Warning ("Deploy binaire impossible pour Nuke {0} (fichier verrouille). Ferme Nuke puis relance le script. Détail: {1}" -f $version, $_.Exception.Message)
                }
            }
        }
        Write-Host ""
        Write-Host "Deploy done: $nukePluginRoot (bin + python/resources)" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "All requested versions built successfully." -ForegroundColor Green
}
finally {
    Pop-Location
}

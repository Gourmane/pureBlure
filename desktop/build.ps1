$ErrorActionPreference = 'Stop'
$studioRoot = Split-Path -Parent $PSScriptRoot
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$source = Join-Path $PSScriptRoot 'Launcher.cs'
$icon = Join-Path $PSScriptRoot 'PureBlur.ico'
$output = Join-Path $studioRoot 'PureBlur Studio.exe'

& $compiler /nologo /target:winexe /optimize+ /reference:System.Windows.Forms.dll "/win32icon:$icon" "/out:$output" $source
if ($LASTEXITCODE -ne 0) { throw 'PureBlur launcher build failed.' }

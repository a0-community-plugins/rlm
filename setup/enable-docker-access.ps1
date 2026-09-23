[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Container,
    [switch]$Apply,
    [switch]$Check,
    [switch]$Yes,
    [string]$Socket = '/var/run/docker.sock'
)
$ErrorActionPreference = 'Stop'
if ($Apply -and $Check) { throw 'Choose -Apply or -Check, not both.' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker command not found. Start Docker Desktop and open a new PowerShell window.'
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop, select Linux containers, and try again.' }
$image = & docker inspect --type container --format '{{.Image}}' $Container
if ($LASTEXITCODE -ne 0) { throw 'Container not found. Copy its name from Docker Desktop.' }
if ($Apply) {
    Write-Host "Set up RLM for: $Container"
    Write-Host 'Agent Zero will restart. Its ports, data and installed software are preserved.'
    Write-Host 'A stopped rollback copy and local snapshot image are kept; never publish the image.'
    Write-Host 'This grants Agent Zero control of Docker and its containers through the Docker socket.'
    if (-not $Yes) {
        $answer = Read-Host 'Continue? [y/N]'
        if ($answer -notmatch '^(y|yes)$') { Write-Host 'Cancelled.'; exit 0 }
    }
}
$arguments = @('run', '--rm', '-i', '--user', '0', '--network', 'none',
    '--mount', "type=bind,source=$Socket,target=/var/run/docker.sock",
    '--entrypoint', '/opt/venv-a0/bin/python3', $image.Trim(), '-',
    '--container', $Container, '--socket', $Socket)
if ($Apply) { $arguments += @('--apply', '--yes') }
# No local Python, WSL, Compose, or Unix socket on Windows is needed.
Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'docker_desktop_setup.py') | & docker @arguments
if ($LASTEXITCODE -ne 0) { throw 'RLM setup failed. See the diagnostic above; do not delete the rollback copy.' }

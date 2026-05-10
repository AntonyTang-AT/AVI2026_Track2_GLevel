# 在 Windows PowerShell 本机执行：从服务器 scp 拉回 artifacts 与常见日志。
#
# 示例:
#   $env:HOST = "183.196.130.56"
#   $env:PORT = "24322"
#   $env:USER = "emo"
#   $env:REMOTE_ROOT = "/home/emo/antonytang/AVI2026_Track2_GLevel"
#   powershell -ExecutionPolicy Bypass -File .\tools\pull_server_artifacts.ps1
#
# 需已安装 OpenSSH 客户端（Windows 可选功能「OpenSSH 客户端」），或 Git 自带的 scp。

param(
    [string]$HostName = $env:HOST,
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 24322 }),
    [string]$User = $(if ($env:USER) { $env:USER } else { "emo" }),
    [string]$RemoteRoot = $(if ($env:REMOTE_ROOT) { $env:REMOTE_ROOT } else { "/home/emo/antonytang/AVI2026_Track2_GLevel" }),
    [string]$LocalBase = $(if ($env:LOCAL_BASE) { $env:LOCAL_BASE } else { ".\server_pull" })
)

if (-not $HostName) { $HostName = "183.196.130.56" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$dest = Join-Path $LocalBase "pull_$ts"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$target = "${User}@${HostName}:${RemoteRoot}"
Write-Host "[pull] $target -> $dest"

$scpArgs = @("-P", "$Port", "-o", "StrictHostKeyChecking=accept-new", "-r", "${target}/artifacts", $dest)
& scp @scpArgs
if ($LASTEXITCODE -ne 0) { Write-Host "[pull] artifacts 目录拉取失败（可能尚未在服务器运行扫描脚本）" }

foreach ($f in @("debug-f0e227.log", "train_glevel.log", "nohup.out")) {
    $one = @("-P", "$Port", "-o", "StrictHostKeyChecking=accept-new", "${target}/$f", $dest)
    & scp @one 2>$null
}

Write-Host "[pull] 完成: $dest"
Get-ChildItem $dest

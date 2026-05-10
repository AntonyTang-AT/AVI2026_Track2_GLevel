# 本机 Windows：使用与工程同级的 server\data 镜像训练（路径见下）
# 用法：在 AVI2026_Track2_GLevel 目录下
#   powershell -ExecutionPolicy Bypass -File .\scripts\glevel_train.local.ps1
# 或先设数据镜像根目录：
#   $env:DATA_MIRROR = "D:\path\to\server\data"

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ($env:DATA_MIRROR) {
    $DataMirror = $env:DATA_MIRROR
} else {
    $DataMirror = Join-Path (Split-Path -Parent $RepoRoot) "server\data"
}
if (-not (Test-Path -LiteralPath $DataMirror)) {
    Write-Error "找不到数据镜像: $DataMirror`n请设置 `$env:DATA_MIRROR 或将 server\data 放在工程上一级目录。"
}

$sl = Join-Path $DataMirror "Super-Lu\dataset"
$featTrain = Join-Path $sl "train_feature"
$featVal = Join-Path $sl "val_feature"
$featTest = Join-Path $DataMirror "AVI2026\test_feature"
$trainCsv = Join-Path $sl "train_data.csv"
$valCsv = Join-Path $sl "val_data.csv"
$ratingCsv = $trainCsv
$testCsv = Join-Path $RepoRoot "data\test_data_basic_information.csv"

foreach ($p in @($featTrain, $featVal, $featTest, $trainCsv, $valCsv, $testCsv)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Error "缺少路径: $p" }
}

Set-Location -LiteralPath $RepoRoot
Write-Host "DATA_MIRROR=$DataMirror"
Write-Host "FEAT_TEST=$featTest"

python (Join-Path $RepoRoot "python\train_task2_glevel.py") `
  --train_csv $trainCsv `
  --val_csv $valCsv `
  --test_csv $testCsv `
  --rating_csv $ratingCsv `
  --labels_in_split_csv `
  --label_col g_level `
  --question q1 q2 q3 q4 q5 q6 `
  --video_dim 512 `
  --video_dir (Join-Path $featTrain "video") `
  --audio_dim 512 `
  --audio_dir (Join-Path $featTrain "audio") `
  --text_dim 768 `
  --text_dir (Join-Path $featTrain "text") `
  --val_video_dir (Join-Path $featVal "video") `
  --val_audio_dir (Join-Path $featVal "audio") `
  --val_text_dir (Join-Path $featVal "text") `
  --test_video_dir (Join-Path $featTest "video") `
  --test_audio_dir (Join-Path $featTest "audio") `
  --test_text_dir (Join-Path $featTest "text") `
  --batch_size 16 `
  --num_epochs 200 `
  --early_stop_patience 30 `
  --lr_scheduler_patience 3 `
  --learning_rate 1e-4 `
  --output_model best_model_glevel.pth `
  --loss_plot_path .\loss_img\loss_glevel.png `
  --test_output_csv (Join-Path $RepoRoot "reports\submissions\submission_glevel.csv") `
  --num_workers 0

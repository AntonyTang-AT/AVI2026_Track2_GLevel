# 本机仅推理 + 导出 submission（路径同 scripts\glevel_train.local.ps1）
#   powershell -ExecutionPolicy Bypass -File .\scripts\glevel_test.local.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ($env:DATA_MIRROR) {
    $DataMirror = $env:DATA_MIRROR
} else {
    $DataMirror = Join-Path (Split-Path -Parent $RepoRoot) "server\data"
}
$sl = Join-Path $DataMirror "Super-Lu\dataset"
$featTrain = Join-Path $sl "train_feature"
$featVal = Join-Path $sl "val_feature"
$featTest = Join-Path $DataMirror "AVI2026\test_feature"
$trainCsv = Join-Path $sl "train_data.csv"
$valCsv = Join-Path $sl "val_data.csv"
$ratingCsv = $trainCsv
$testCsv = Join-Path $RepoRoot "data\test_data_basic_information.csv"
$model = if ($env:TEST_MODEL) { $env:TEST_MODEL } else { ".\best_model_glevel.pth" }
$textDim = if ($env:TEXT_DIM) { $env:TEXT_DIM } else { "768" }
$outCsv = if ($env:TEST_OUTPUT_CSV) { $env:TEST_OUTPUT_CSV } else { (Join-Path $RepoRoot "reports\submissions\submission_glevel.csv") }

Set-Location -LiteralPath $RepoRoot

python (Join-Path $RepoRoot "python\train_task2_glevel.py") `
  --only_test `
  --test_model $model `
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
  --text_dim ([int]$textDim) `
  --text_dir (Join-Path $featTrain "text") `
  --val_video_dir (Join-Path $featVal "video") `
  --val_audio_dir (Join-Path $featVal "audio") `
  --val_text_dir (Join-Path $featVal "text") `
  --test_video_dir (Join-Path $featTest "video") `
  --test_audio_dir (Join-Path $featTest "audio") `
  --test_text_dir (Join-Path $featTest "text") `
  --batch_size 16 `
  --num_workers 0 `
  --test_output_csv $outCsv

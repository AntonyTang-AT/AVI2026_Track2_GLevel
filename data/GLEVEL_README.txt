如何得到 glevel_train.csv（认知能力三分类标签）
============================================

一、正确来源（比赛/科研）
------------------------
g_level 必须是官方或赛方提供的标注，不能凭空编造。
常见形式：一张表含「被试 id」+「Low/Medium/High」或整数标签。
整数标签须与训练参数一致：**0/1/2**（0=Low）为默认；若赛方使用 **1/2/3**（1=Low），可加 **`--g_level_int_encoding one`**。
当默认 zero 时，若 train+val 中**纯整数标签出现 3 且从未出现 0**，脚本会**自动按 1–3 解释为 one**（并打印提示）；若仅含 1、2 而无 0/3，仍保持 zero，请自行确认赛方约定。
训练开始时会打印「原始取值计数」与「编码后」类频，请核对是否与官方分布一致。

1) 若赛方给你一个 CSV/Excel，列名可能是 id、g_level，或 id、Cognitive_Level 等：
   - 先看列名：
       python tools/peek_csv_header.py /路径/表.csv
   - 从该表生成训练用子集（只保留 train+val 里的 id）：
       python tools/build_glevel_train_csv.py \
         --source /路径/官方标签表.csv \
         --id_col id \
         --label_col g_level \
         --train_csv ./data/train_data.csv \
         --val_csv ./data/val_data_new.csv \
         --out ./data/glevel_train.csv
   若标签列不叫 g_level，把 --label_col 改成实际列名（输出文件里仍会统一成 g_level）。

2) 若官方把 g_level 合并进了 all_data.csv：
   - 无需单独 glevel_train.csv，训练时可去掉 --glevel_csv，并保证 --rating_csv 指向含 g_level 的表。

二、train 有 g_level、val 没有（常见）
------------------------
训练同时加：

  --labels_in_split_csv
  --glevel_csv ./data/glevel_val_supplement.csv

其中 glevel_val_supplement.csv 只需包含「验证集 id + g_level」（也可用 train+val 全集，代码只取当前划分里的 id）。
可用 tools/build_glevel_train_csv.py 从赛方 val 标签表生成。

三、train.csv / val.csv 里均已有 g_level
------------------------
仅加：

  --labels_in_split_csv

无需 glevel_train.csv。vote_train_glevel.sh 默认 SPLIT_LABELS=1。

**多模态（与常见 ~51% 基线对齐，非 text_gru）**：历史上该量级多来自 **音+视+文 SharedMLPwEnsemble + 赛方 train/val 表**。若要与旧数可比，请使用赛方 **`train_data.csv` / `val_data.csv`**（勿混用 `train_fixed`），并在项目根执行 **`bash vote_train_glevel_multimodal.sh`**（内置 `cross_modal_attn`、dropout、label_smoothing、`balanced_acc` 选模、固定 seed 与长早停等；默认产出 `best_model_glevel_multimodal_plus.pth`）。训练前若 shell 里残留过 `GLEVEL_OPT='--glevel_arch text_gru ...'`，请先 **`unset GLEVEL_OPT`**。可选 `export MM_TEMPORAL=1`、`export MM_MEDIUM_BOOST=1`（温和过采样 Medium，见 `tools/glevel_plan_runbook.txt` 阶段二）、`export RUN_TEST_AFTER=1`（测试特征齐全时连跑 `vote_test_glevel.sh`），详见脚本顶部注释。**阶段性数据与问题拆解**：`tools/GLEVEL_PHASE_REPORT_zh.txt`。

**计划书实验脚本（`tools/`，均需项目根、与 vote 相同 export）**：
- `bash tools/run_ablation_temporal_multimodal.sh`：对照是否加 **6 题时序 GRU**（日志在 `logs/ablation_temporal/`）。
- `bash tools/run_ablation_select_best_multimodal.sh`：`balanced_acc` / `macro_f1` / `val_ce` 选模对照。
- `bash tools/run_multi_seed_multimodal.sh`：多 seed 训练并汇总 `logs/multi_seed_mm/metrics_seeds_*.csv`（依赖日志行 **`[metrics_line]`**）。
- `bash tools/run_kfold_multimodal_plus.sh`：与多模态 preset 一致的 **K 折 + 提交投票**（`KFOLDS`/`KFOLD_OUT_DIR` 等同 `one_click_kfold_glevel.py`）。
- **重要**：K 折 / 多种子与单折训练共用 **`FEAT_TRAIN` / `FEAT_VAL` / `FEAT_TEST`、`TEXT_*`**。说明文档里的 **`/path/to/...` 与 `export FEAT_TRAIN="..."` 仅为示意**；若把字面量 `"..."` 写进 shell，会变成路径 `.../audio`（不存在），日志将出现 **「特征不完整，已剔除 411/411」** 或 **450/450**。请复制你**已能跑通** `bash vote_train_glevel_multimodal.sh` 时的整组 `export`，再跑上述脚本。
- `bash tools/preflight_test_submission.sh`：仅测 **测试集** audio/video + 文本 `.npy` 覆盖，便于打通 `predict_test`。
- `bash tools/run_glevel_coral_multimodal.sh`：**CORAL 序关系损失**（`train_task2_glevel.py --glevel_loss coral`，三分类专用）。**`vote_test_glevel.sh` 加载 CORAL 权重时须在 `GLEVEL_OPT` 中含 `--glevel_loss coral`**，否则最后一层维数不匹配。
- `bash tools/run_multimodal_medium_focus.sh`：**Medium / macro-F1** 预设（manual 类权 `1,2,1` + 关平衡采样 + `select_best=macro_f1`）；**勿再叠 `--class_weight auto`**。

**序数 / 跨档错分（实验顺序）**：优先对比 **`--glevel_loss ce` vs `coral`**（CORAL 已为 K−1 辅助二分类，与「单标量+阈值」ordinal 不同）；单标量 ordinal 头暂缓，待 CORAL 调参后再评估是否值得增加维护成本。

**`train_task2_glevel.py` 扩展（与多模态 / `--temporal_gru` 配合）**：
- **`--temporal_bidirectional`**：双向 GRU，输出经 Linear 回到 `fused_dim`。
- **`--temporal_attn_pool`**：对 GRU 逐题输出做 softmax 注意力加权（替代 `mean`/`last`）。
- **`--question_pos_embed_dim D`**：每题可学习嵌入 + Linear 压回 `fused_dim`；`0` 关闭。
- **`--temporal_step_dropout_p`**：按时间步随机整步置零（训练时）。
- **`--swa_start_epoch N`** + **`--swa_lr`**：第 N 个 epoch 起 SWA；结束后 **`update_bn`**，权重写入 **`--output_swa_model`**（默认 `output_model.swa.pth`），并打印 **`[swa_eval]`**。
- **`--tta_times` / `--tta_noise_std`**：`only_test` 与训练结束 **`predict_test`** 时对特征加噪多次平均概率（0 关闭）。
- **`--optim sam --sam_rho 0.05`**：SAM（底层 AdamW，每步约两倍计算）。

Shell：`export MM_BIDIRECTIONAL=1` 且 **`MM_TEMPORAL=1`** 时，`vote_train_glevel_multimodal.sh` 会追加 `--temporal_bidirectional --temporal_attn_pool`。

四、仅跑通代码（假标签，无意义）
------------------------
没有真实标签时，只能测试「代码能否跑完」：

   python tools/glevel_dummy_for_smoke_test.py --out ./data/glevel_train.csv

得到的模型不能用于提交或报告。

五、单独 glevel_csv（与划分无关）
------------------------
   --glevel_csv ./data/glevel_train.csv

六、5 折集成与 Nanbeige 文本特征（可选）
------------------------
- 合并 train+val 做分层 K 折并多数投票融合提交（推荐，无 CRLF 问题）：
    cd /path/to/AVI2026_Track2_GLevel
    python one_click_kfold_glevel.py
  或：bash vote_kfold_glevel.sh（项目根，内部 `python one_click_kfold_glevel.py`）
  或手动：python tools/run_kfold_glevel.py --merge ... --  --text_dim 768
  再：python tools/ensemble_glevel_csv.py --inputs kfold_glevel_out/fold0_submission.csv ... --out submission_glevel_kfold_vote.csv
- 合并折训练时，部分 id 仅在 val_feature 下有 .npy，训练脚本会自动加 --train_feat_fallback（由 run_kfold_glevel 在 --merge 时传入）。
- 重提文本向量（推荐）：export TEXT_ROOT=... OUT_ROOT=... 后执行
    python tools/extract_nanbeige_one_click.py
  或 bash tools/extract_nanbeige_text.example.sh
  以 stderr 打印的维数设置 TEXT_DIM 与 TEXT_TRAIN_DIR 等（见 vote_train_glevel.sh）。**`vote_test_glevel.sh` 已与训练脚本对齐**：同样识别 `NANBEIGE_TEXT` / `NANBEIGE_TEXT_SUBDIR`，推理前请 export 与训练一致。

七、文本提取卡住 / 试跑 / 后台（可选）
------------------------
- 终止：终端按 Ctrl+C；或另开终端执行  pkill -f "features.extract_text"
- 先试跑少量文件（不下载完也可先验证流程）：  export MAX_FILES=30
- 换更小权重（下载更小）：  export MODEL_ID=Nanbeige/Nanbeige4-3B-Base
- 后台写日志：  nohup python tools/extract_nanbeige_one_click.py > extract_nb.log 2>&1 &
- extract_text 支持 --max_files；训练侧可选 --class_weight auto 与 --label_smoothing 0.05（见 train_task2_glevel.py --help）。**MixUp 默认关闭**（`--mixup_prob 0`）。`train_task2_glevel.py` 默认早停耐心 **40**，并支持 **`--early_stop_min_epochs`**（先训满若干 epoch 才允许早停）。**`vote_train_glevel.sh` 在命令行末尾**传入默认 `--early_stop_patience 40`、`--early_stop_min_epochs 12`、`--lr_scheduler_patience 5`（可用环境变量 **`EARLY_STOP_PATIENCE` / `EARLY_STOP_MIN_EPOCHS` / `LR_SCHEDULER_PATIENCE`** 覆盖，且会覆盖 `GLEVEL_OPT` 里同名参数）。仍过早停时可继续加大耐心或换 `--select_best`。
- 时序：可加 --temporal_gru（6 题 fused 序列经 GRU 再分类），默认关闭。
- 鲁棒：--modality_dropout_p 0.1~0.2 训练时随机丢一整模态（eval 不丢）。
- 融合：--cross_modal_attn 在拼接前对 video/text/audio 做一层（或多层 --cross_modal_layers）Transformer 自注意力；可与 --temporal_gru 同开。
- **`import torch` 报 `undefined symbol: nccl...`**：属 **PyTorch CUDA 构建与系统 NCCL/CUDA 运行时栈不一致**（与仓库训练逻辑无关）。请在 **新 conda 环境** 按 https://pytorch.org 选择与节点 CUDA 匹配的命令重装 `torch`/`torchvision`/`torchaudio`，或暂用 **CPU 版** wheel；勿强依赖已损坏的 base。训练/测试脚本支持指定解释器：**`export PYTHON=/path/to/conda/envs/你的环境/bin/python`** 后再执行 `bash vote_train_glevel.sh` / `bash vote_test_glevel.sh` / `bash vote_kfold_glevel.sh`。`train_task2_glevel.py` 在导入失败时会打印简要排查提示。
- 失败时项目根会生成/追加 **`debug-f0e227.log`**（`vote_*` 预检或手动 `python tools/diagnose_torch_env.py`）。跳过 shell 预检：`export SKIP_TORCH_PREFLIGHT=1`（仍会在 `train_task2_glevel.py` 首行 `import torch` 处失败并写日志）。
- 预检失败时终端会打印 **`tools/print_torch_env_fix_hint.sh`** 中的**可复制** conda/pip 示例；也可单独执行：`bash tools/print_torch_env_fix_hint.sh`。
- **根本规避 NCCL/base 混装**：**`python3 tools/bootstrap_isolated_cpu_env.py`**（推荐，不受 bash CRLF 影响）在项目根创建 `.venv_glevel_cpu` 并装 CPU 版 torch；再 `source .venv_glevel_cpu/bin/activate`、`export PYTHON=.../bin/python` 后跑 `vote_train_glevel.sh`。备选：`bash tools/bootstrap_isolated_cpu_env.sh`。若 shell 脚本在 Linux 报 `set: pipefail`，对仓库执行 `git add --renormalize .` 或 `sed -i 's/\\r$//' tools/*.sh *.sh`。训练走 CPU、较慢，但与损坏的 `(base)` 隔离。

八、类别不平衡（避免「全猜 High」或「全猜 Medium」塌缩）
------------------------
默认：**`WeightedRandomSampler`**（batch 内近似类平衡）+ **`class_weight=none`**（均匀 CE）。
「采样已拉平少数类」时再开 **`--class_weight auto`** 会双重点少数类，易把预测整批吸到 Medium；确需叠加时请自行观察 `val_pred_classes`。
若两者同时启用，**`train_task2_glevel.py` 启动时会打印 WARNING**；仍建议只保留一种重加权方式。

关闭平衡采样：`--no_balanced_sampler`。仅 CE 加权、不要采样：`--no_balanced_sampler --class_weight auto`。

Nanbeige 文本（2560）：全量提取到 `train_feature/text_nb` 后 **`export NANBEIGE_TEXT=1`**。试跑 **`NANBEIGE_TEXT_SUBDIR=text_nb_smoke`** 时，脚本会把 val/test 文本目录默认指到与 train 相同的 smoke 目录；**官方 val id 若未出现在该目录的 .npy 中，验证集仍会全部被剔除**——需对验证集转写单独跑 extract，`OUT` 指向 `val_feature/text_nb_smoke` 等，并 **`export TEXT_VAL_DIR=...`**。

保存 best 默认 **`macro_f1`**（仅 val 出现类）；平局比 val CE。改：`--select_best val_ce`。日志含 **`val_pred_classes`**（验证预测用了几类）。

九、验证集错分分析（改进模型前先看错哪类）
------------------------
训练结束会用 best checkpoint 再跑一遍验证；若需逐样本对照，加：

  --val_errors_csv ./logs/val_glevel_errors.csv

或：  export VAL_ERRORS_CSV=./logs/val_glevel_errors.csv  后 bash vote_train_glevel.sh

写出 CSV：id、真值/预测、各类 prob、CE、是否正确；并合并 val 划分表中的列（便于对照被试元数据）。
终端打印：混淆矩阵、各类 recall、错分「真→预测」计数（如 Medium→High）。
CSV 中的 **`prob_class0/1/2`** 与 **`margin_top2`**（Top2 概率差）：若几乎所有样本 **`prob_class1` 最高** 且 margin 很小，多为预测塌缩或特征判别力弱；可与日志里 **`val_pred_classes`**、**连续 epoch 单类 WARNING** 对照。

若验证集上 **连续多 epoch `val_pred_classes=1`**，脚本会打出排查提示（标签编码、双重点权、`--val_errors_csv`、特征）。

对已训模型单独分析（路径须与训练一致，含 GLEVEL_OPT）：

  export VAL_ERRORS_CSV=./logs/val_glevel_errors.csv
  bash vote_test_glevel.sh

或一键（同上，默认写出 `./logs/val_glevel_errors.csv`）：

  bash tools/run_val_error_analysis.sh

或手写：

  python train_task2_glevel.py --only_test --test_model best_model_glevel.pth \\
    ... 同 vote_train_glevel 的特征与 csv 参数 ... \\
    --val_errors_csv ./logs/val_errors.csv

十、服务器路径扫描与结果拉回本机（可选）
------------------------
**1）在服务器项目根执行扫描**（汇总环境变量、默认/实际 CSV 与特征路径、各模态 `.npy` 数量、`torch` 与 `FEATURE_LOADER_REVISION`、并拼接常见日志尾部）：

  cd /path/to/AVI2026_Track2_GLevel
  export PROJECT_ROOT="$PWD"   # 可选
  python3 tools/server_environment_scan.py
  # 附加抓取: python3 tools/server_environment_scan.py --tail-lines 400 --capture-glob "logs/*.log"

生成文件在 **`artifacts/`**：`server_scan_latest.json`、`server_scan_latest.txt`、`captured_logs.txt`。

**2）在本机拉回**（需 OpenSSH 的 `scp`，Windows 可装「OpenSSH 客户端」或用 Git Bash）：

- Linux / macOS / Git Bash:

    export HOST=183.196.130.56 PORT=24322 USER=emo
    export REMOTE_ROOT=/home/emo/antonytang/AVI2026_Track2_GLevel
    bash tools/pull_server_artifacts.sh

- Windows PowerShell:

    $env:HOST="183.196.130.56"; $env:PORT="24322"; $env:USER="emo"
    $env:REMOTE_ROOT="/home/emo/antonytang/AVI2026_Track2_GLevel"
    powershell -ExecutionPolicy Bypass -File .\tools\pull_server_artifacts.ps1

本机默认落到 **`./server_pull/pull_<时间戳>/`**（含 `artifacts/` 副本及若能拉到的 `debug-f0e227.log` 等）。`artifacts/` 与 `server_pull/` 已加入 `.gitignore`。

十一、验证集补特征与冲分检查清单（目标更高 acc / 官方榜）
------------------------
**0）路线 A 一键预检（推荐训练或 K 折前执行）**

与 `vote_train_glevel.sh` 相同地 `export TRAIN_CSV`、`VAL_CSV`、`TEST_CSV`、`FEAT_*`、`NANBEIGE_TEXT`、`TEXT_*_DIR` 等后：

  bash tools/route_a_complete.sh

或训练时自动跑预检：

  export ROUTE_A_PREFLIGHT=1
  bash vote_train_glevel.sh

步骤包括：`FEAT_TEST` 的 audio/video（`NANBEIGE_TEXT=1` 时不强制 `FEAT_TEST/text`）、val 三模态缺失报告、val/test 的 **文本** `.npy` 全覆盖检查。未通过时按终端提示补特征或见 `tools/extract_nanbeige_splits.example.sh`。

**1）某条 val id 缺 .npy，验证集少 1 条、指标方差大**

与训练相同的 primary + fallback 规则下，逐行打印「缺哪一题、哪一模态」：

  python tools/report_missing_features_for_csv.py \\
    --csv ./data/val_data_new.csv \\
    --audio_dir "$FEAT_VAL/audio" --video_dir "$FEAT_VAL/video" --text_dir "$TEXT_VAL_DIR" \\
    --fallback_audio_dir "$FEAT_TRAIN/audio" --fallback_video_dir "$FEAT_TRAIN/video" \\
    --fallback_text_dir "$TEXT_TRAIN_DIR"

补提对应 audio/video/text 向量后，再训练；退出码 2 表示存在不完整 id。

**2）全量 Nanbeige 文本 + 三划分一致**

- 对 **train / val / test** 转写分别或统一提取到各 `TEXT_*_DIR`，避免 smoke 目录只有 train id 导致 val 全剔除。
- **勿**把 `TEXT_VAL_DIR` / `TEXT_TEST_DIR` 指到「仅含 train id」的目录（例如只移了训练集 `text_nb`）：验证集/测试集 id 与训练不同，主目录里若没有 `{该id}_{q}.npy`，会整表被剔除并报 `val_data.csv 过滤后无剩余样本`。应对：对 val（及 test）转写再跑 `extract_nanbeige_one_click.py`，`OUT_ROOT` 指到含 **对应划分 id** 的目录（可与 train 合并到同一 `text_nb` 目录以增加文件，或分 `text_nb_val` / `text_nb_test` 再分别 `export`）。
- `export NANBEIGE_TEXT=1`，`TEXT_DIM=2560`，`vote_train_glevel.sh` / `vote_test_glevel.sh` 与 K 折脚本中路径一致。

**3）GPU 与解释器**

有 GPU 时使用与 CUDA 匹配的干净环境（见上文 `import torch` / NCCL 说明）；`export PYTHON=/path/to/env/bin/python` 后跑 `vote_train_glevel.sh` / `vote_kfold_glevel.sh`。

**4）超参与结构消融（`GLEVEL_OPT`）**

- 选模：`--select_best macro_f1`（默认）、`balanced_acc`、`val_ce` 对比。
- 结构：依次试 `--temporal_gru --temporal_pool mean`、`--cross_modal_attn`、`--modality_dropout_p 0.12`（可与关 MixUp、长早停组合）。
- 详见 `vote_train_glevel.sh` 顶部注释示例。

**5）K 折 + 提交融合**

  bash vote_kfold_glevel.sh

或（跑 K 折**前**先检查文本；可加全量路线 A：`export ROUTE_A_PREFLIGHT=1`）：

  bash tools/run_kfold_glevel_submit.sh

`one_click_kfold_glevel.py` 已与 `vote_train_glevel.sh` 对齐：设 **`export NANBEIGE_TEXT=1`** 时默认使用 `FEAT_*/text_nb`（或 `NANBEIGE_TEXT_SUBDIR`、`*smoke*` 规则），并尊重已 `export` 的 `TEXT_TRAIN_DIR` / `TEXT_VAL_DIR` / `TEXT_TEST_DIR`。

输出目录默认 `KFOLD_OUT_DIR=./kfold_glevel_out`，融合文件 `submission_glevel_kfold_vote.csv`。亦可手动：

  python tools/ensemble_glevel_csv.py --inputs fold0.csv fold1.csv ... --out submission_merged.csv

**6）仅检查「文本 .npy」是否齐（train/val/test 任一划分）**

  python tools/check_text_npy_coverage.py --csv "$TEST_CSV" --text_dir "$TEXT_TEST_DIR" \\
    --fallback_text_dir "${TEXT_TRAIN_DIR:-}"

退出码 2 表示有 id 缺题；训练/推理前对 val、test 各跑一次可减少「过滤后无剩余样本」。

**7）Nanbeige 按划分提取示例（OUT 勿写只读 Super-Lu 时）**

见 `tools/extract_nanbeige_splits.example.sh`（注释模板）。

**8）消融 `GLEVEL_OPT` 预设文案**

见 `tools/glevel_ablation_presets.sh`（打开复制 export 行）。

十二、路线 D：类权与采样（针对 Medium，勿双叠）
------------------------
默认 **WeightedRandomSampler + class_weight=none**。若希望 **仅 CE 类频加权**、不用平衡采样：

  export GLEVEL_OPT="--no_balanced_sampler --class_weight auto"

**勿**在保留默认平衡采样的同时再加 `--class_weight auto`（README 第八节 WARNING）。可先单折 smoke 观察 `val_pred_classes` 与 `[val_summary]` 是否塌缩。

十三、第二阶段（可选代码，当前仓库未默认实现）
------------------------
若 CLI 消融与 K 折仍不足，可再评估：**Focal loss**、**SWA/EMA**、**减小 M_model ensemble 宽度或冻结 adapter**（需改 `model/vote_model/M_model.py` 与训练循环）。属高成本实验项。

十四、改进计划书（阶段一至四，不依赖测试集）
------------------------
- **重划分验证**：`python tools/split_train_val.py` 生成 `train_fixed.csv` / `val_fixed.csv`（分层默认 **15%**；更小训练集可用 `--val_ratio 0.10`，更大验证集可用 `--val_ratio 0.2`）。
- **新训练开关**：`train_task2_glevel.py` 支持 `--glevel_arch text_gru|text_mlp|shared_mlp`、`--class_weight manual`、`--sampler_medium_boost`、`--weight_decay`、`--scheduler_min_lr`、`--lr_scheduler cosine|plateau`、`--mlp_dropout`、`--seed` 等。
- **命令模板**：见 `tools/glevel_plan_runbook.txt`。
- **说明**：验证准确率须在你方机器上训练后从日志 `[val_summary]` 读取；本仓库无法在 Cursor 内代跑 GPU/服务器实验。

# AVI2026_Track2_GLevel

基于 **ACM MM 2025 AVI Track2 冠军方案（365Aspects / HFUT-VisionXL）** 的多模态骨架，面向 **AVI 2026 赛道二 `g_level` 三分类（认知能力等级）** 的训练、扫参、提交与辅助工具代码库。

**GitHub**：<https://github.com/AntonyTang-AT/AVI2026_Track2_GLevel>  
**SSH**：`git@github.com:AntonyTang-AT/AVI2026_Track2_GLevel.git`

---

## 功能概览

| 模块 | 说明 |
|------|------|
| **训练** | `python/train_task2_glevel.py`：多模态 G-Level 主训练（验证指标、早停、可选交叉模态注意力等）。 |
| **推理 / 提交** | `python/predict_submission.py`：按 checkpoint 与特征目录生成测试集 `submission.csv`。 |
| **数据管线** | `dataset/baseline_dataset2_vote.py`：读取 CSV + 音频/视频/文本 `.npy` 特征；标签工具 `dataset/glevel_labels.py`。 |
| **Shell 入口** | `scripts/glevel_train.sh`、`scripts/glevel_test.sh`、`scripts/glevel_train_multimodal.sh`、`scripts/glevel_kfold.sh`、`scripts/glevel_smoke_one_click.sh`；原 Track2 回归见 `scripts/track2_train.sh`。 |
| **GPU 扫参** | `tools/run_glevel_gpu_combo_sweep.sh`：多组超参 × 多种子；可选合并 train+val 池（`POOL_RANDOM_SPLITS`）或稳定性划分（`PARTITION_ROUNDS`，与 pool 互斥）。 |
| **伪标签 / 集成** | `tools/build_*pseudo*.py`、`tools/ensemble_glevel_csv.py`、`external/submissions_peer/` 等。 |
| **DeepSeek 标注** | `python/annotate_with_deepseek.py` 等；默认 JSON 输出目录 **`reports/deepseek/`**。 |
| **启发式测试恢复** | `heuristic_test_recovery/`（非赛方官方流程）。 |

更细的产出路径说明见 `reports/DATA_LABEL_PATHS.txt`。

---

## 环境依赖

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 或精简依赖：
# pip install -r requirements-core.txt
```

GPU 训练需本机安装与 **CUDA 匹配的 PyTorch**（版本随机器而定）。脚本里可通过 **`GLEVEL_CUDA_PYTHON`** 指定用于训练的 Python 可执行文件。

---

## 数据与路径约定

### 1. 赛事完整数据根目录（特征体积大，不在 Git 中）

默认期望本机存在赛方数据树，并通过环境变量指向根目录：

```bash
export SUPERLU_DATASET=/data/Super-Lu/dataset   # 按你的机器修改
```

其下通常包含官方 `train_data.csv`、`val_data.csv`、`test_data_basic_information.csv` 以及 `train_feature/`、`val_feature/` 等（体量可达十余 GB）。训练脚本与 `tools/glevel_paths.inc.sh` 等均以该变量为默认参照。

### 2. 仓库内已镜像的官方 CSV（可提交、体积小）

目录 **`data/superlu_official/`** 内为从 `SUPERLU_DATASET` 同步的 **官方标签 CSV** 副本，便于在无 `/data/Super-Lu` 的机器上对齐列表与划分：

```bash
bash tools/sync_superlu_official_csv.sh
```

### 3. 仓库内随代码提供的特征子集

**`data/`** 下还包含部分 **文本/测试特征 `.npy`**（如 `text_nb/`、`test_feature/` 等），用于 NanBeige 文本管线或本地冒烟；**完整训练特征仍需指向 `SUPERLU_DATASET`**。

### 4. 不会进入 Git 的内容（见 `.gitignore`）

- `experiments/`、`checkpoints/`、`*.pth` / `*.pt`：实验输出与权重。  
- `kfold_glevel_out/`、`kfold_glevel_multimodal_plus/`、`smoke_check/`、`.svm_smoke_out/`：大规模或可重建产物。  
- `.venv*/`：虚拟环境。  
- `logs/*.json`、`args_log/*.json`：含绝对路径的运行快照（目录保留 `.gitkeep`）。  
- `deepseek_annotations*.json` 等敏感或大体积标注缓存。

克隆仓库后需 **自行训练或从本机拷贝权重**，不可指望远程自带 checkpoint。

---

## 快速上手

### 单次训练（示例）

推荐：在仓库根目录执行 **`bash scripts/glevel_train.sh`**（内部调用 `python/train_task2_glevel.py`）。或直接：

```bash
python python/train_task2_glevel.py --help
```

### GPU 组合扫参（含合并 train+val 池 + 多随机划分）

```bash
export SUPERLU_DATASET=/path/to/dataset
export HUNT_DIR=./experiments/gpu_combo_sweep/my_pool_run
export POOL_RANDOM_SPLITS=5      # 合并池后随机划分次数（slot）
export POOL_TRAIN_N=418           # 每轮抽作训练的条数，其余作验证
export SAVE_TOP_K_MODELS=10
bash tools/run_glevel_gpu_combo_sweep.sh
```

后台封装示例：`bash tools/run_pool_merged_wide_background.sh`。

### 稳定性划分 / CV 变体

- `PARTITION_ROUNDS`：与 `POOL_RANDOM_SPLITS` **不要同时非 0**。  
- `tools/run_glevel_gpu_combo_sweep_cv.sh`：交叉验证式扫参。

### 生成提交 CSV

```bash
python python/predict_submission.py --help
```

具体参数需与你的 checkpoint、`test_csv`、特征目录一致。

---

## 项目目录结构（当前仓库）

根目录仅保留 **`README.md`、`LICENSE`、`requirements*.txt`、配置与说明**；代码与入口按目录归类：

```text
AVI2026_Track2_GLevel/
├── README.md / LICENSE / requirements.txt / requirements-core.txt
├── .gitignore / .gitattributes
│
├── python/                         # Python 主入口（请在仓库根运行）
│   ├── train_task2_glevel.py       # G-Level 训练
│   ├── train_task2_vote.py         # 原 Track2 回归训练
│   ├── predict_submission.py       # 推理写出 submission
│   ├── one_click_kfold_glevel.py   # K 折编排
│   ├── annotate_with_deepseek.py
│   └── annotate_deepseek_interactive.py
│
├── scripts/                        # Bash / PowerShell 便捷入口（命名统一 glevel_* / track2_*）
│   ├── glevel_train.sh / glevel_test.sh / glevel_kfold.sh
│   ├── glevel_train_multimodal.sh
│   ├── glevel_smoke_one_click.sh
│   ├── track2_train.sh / track2_test.sh
│   └── glevel_train.local.ps1 / glevel_test.local.ps1
│
├── dataset/
│   ├── baseline_dataset2_vote.py
│   └── glevel_labels.py
├── model/vote_model/…
├── features/…
├── data/（含 superlu_official/ 官方 CSV 镜像）
├── reports/
│   ├── deepseek/                   # DeepSeek JSON / 缓存（纳入版本控制的归档）
│   ├── submissions/               # 示例 submission CSV
│   └── DATA_LABEL_PATHS.txt
├── tools/                          # 扫参、划分、评估、伪标签等（大量 .sh / .py）
├── external/
├── heuristic_test_recovery/
├── background_figs/
├── loss_img/
├── args_log/ / logs/（仅 .gitkeep 提交）
└── train_print_log/
```

默认提交与训练写出路径示例：**`reports/submissions/submission_glevel.csv`**（可用环境变量 `TEST_OUTPUT_CSV` 覆盖）。

本地 **`experiments/`**、**`kfold_*`**、根目录 **`best_model*.pth`** 等为运行产物，默认 **不进入 Git**。

---

## 许可证与致谢

- 本仓库代码许可见 **`LICENSE`**（MIT）。  
- 研究用途请遵守赛题数据协议与主办方规则。  
- 原冠军方案论文与引用保留如下（英文摘要）。

### Listening to the Unspoken: Exploring "365" Aspects of Multimodal Interview Performance Assessment

**[MM 2025]** Official implementation for the ACM Multimedia AVI Challenge 2025 Track 2 championship solution (HFUT-VisionXL).  
原任务为多输入多标签 **回归**（多项职业能力评分）；本仓库在相同骨架上扩展了 **g_level 三分类** 与相关实验脚本。

**Acknowledgments**：AVI 2025 organizers；[MERtools](https://github.com/zeroQiaoba/MERTools) 等开源工具。

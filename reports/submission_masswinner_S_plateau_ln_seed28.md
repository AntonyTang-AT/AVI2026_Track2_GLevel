# 模型报告：`submission_masswinner_S_plateau_ln_seed28`

本文面向「读的人不一定天天盯代码」的写法：**先把这条模型在干什么说清楚，再用一张总表把 train / val / test 数字摆在一起，最后解释为什么验证很高、官方测试却落到约 47%。**

---

## 1. 这条提交是什么、从哪来

| 项 | 内容 |
|----|------|
| 提交 CSV | `external/submissions_peer/submission_masswinner_S_plateau_ln_seed28.csv` |
| 训练 checkpoint | `experiments/gpu_combo_sweep/mass_20260509_145045/S_plateau_ln/seed28/best.pth` |
| 训练日志 | `experiments/gpu_combo_sweep/mass_20260509_145045/S_plateau_ln/seed28/train.log` |
| 训练参数快照（完整） | `logs/args_glevel_20260509_150130.json` |
| 仅推理阶段快照 | `logs/args_glevel_20260509_172551.json` |
| 网格命名 | **combo**=`S_plateau_ln`，**seed**=`28`；搜索目录 **`mass_20260509_145045`** |

**流水线**：同一轮 GPU 上大量「组合 × 随机种子」训练完成后，脚本 **`mass_winners_refresh_iter2`** 按验证集表现挑出优等 checkpoint，写出 `external/submissions_peer/` 下的 peer 提交文件（详见 `experiments/glevel_improvement_plan/mass_winners_refresh_iter2.log`）。

---

## 2. 建模思路（拆成「输入 → 结构 → 怎么训 → 怎么选模型」）

### 2.1 要解决什么问题

赛题一侧给出每位受试者在 **6 道题（q1–q6）** 上的 **音频、视频、文本** 三种模态特征（已是抽取好的向量，不是原始音视频）。模型要做的是：**综合这些特征，预测一个三类的 `g_level`（低 / 中 / 高）**。  
可以把它想成：每个样本是一条很长的「多模态时间线/题目线」，网络在最后吐出一个三选一标签。

### 2.2 输入与文本路线（Nanbeige）

- **音频 / 视频**：维度各 **512**，来自赛方特征目录（训练用 `train_feature`，验证用 `val_feature`，测试用 `test_feature`）。  
- **文本**：不用传统 768 维句向量，而是用 **Nanbeige 2560 维** 表征；工程内路径分别为 **`data/text_nb`**（训练）、**`data/text_nb_val`**（验证）、**`data/test_nb`**（测试），与 `scripts/glevel_train.sh` 一类脚本一致。

这样做的直觉是：**文本信息量很大**，若只用短向量容易丢语义；Nanbeige 维数更高，与音视频拼接后，后面的融合模块有更多「可调空间」。

### 2.3 网络长什么样（组合名 `S_plateau_ln` 在说什么）

名字可以拆开读：

| 片段 | 含义（直觉） |
|------|----------------|
| **S** | **Shared MLP**：三道模态都进同一个多模态骨干（不是「只要音频+文本」的消融版）。 |
| **plateau** | 学习率用 **`ReduceLROnPlateau`**：验证损失一段时间不降，就把学习率乘一个系数往下压；适合「先快后稳」地微调，而不是固定 cosine 曲线一刀切。 |
| **ln** | 打开 **`fused_layer_norm`**：在深层表示上做归一化，减轻尺度漂移，和多模态堆叠一起训练时通常更稳。 |

在此基础上还叠加了竞赛里常用的正则：**`mlp_dropout=0.25`**、**`modality_dropout_p=0.12`**（训练时随机遮一点模态，逼模型别绑死在某一路信号上），以及一层 **跨模态注意力**（`cross_modal_attn`，`cross_modal_layers=1`）：让某一题的 audio / video / text 之间可以互相「看一眼」再往下传。

### 2.4 损失函数与优化器（模型怎么学）

- **优化器**：AdamW；**初始学习率 `1e-4`**；**权重衰减 `1e-3`**。  
- **分类损失**：带 **`label_smoothing=0.05`** 的交叉熵（标签不是绝对 0/1，防止对训练集过于自信）。  
- **类别不均衡**：没有在 loss 里再加 `class_weight`，而是用 **`WeightedRandomSampler`**：按类别频次做反比抽样，并且 **`sampler_medium_boost=1.5`** —— 直觉上 **中间档（Medium）会被多抽一点**，避免模型只会猜多数类。

### 2.5 「存哪一个 checkpoint」——这也是最容易误解的一点

训练脚本每个 epoch 都会在验证集上算一堆指标，但**磁盘上只保留「你认为最重要的那个指标」最好的那次**。  
本模型配置的是：**`select_best = balanced_acc`（验证集均衡准确率）**，不是单纯 `val_acc`，也不是赛方榜单上的测试准确率。

**通俗理解**：

- **普通准确率**：猜对多少条算多少条；若某一类特别多，「全猜那一类」也可能看起来不差。  
- **均衡准确率**：三类各自算召回再平均，更惩罚「只会猜一类」的模型。

因此：这条模型是 **「在验证集上让三类更均衡地做对」** 的方向被选出来的；它和 **「在 130 条测试集上整体命中率最高」** 并不自动等价。

### 2.6 训练跑了多久、停在哪

- 上限 **200** epoch；**早停 patience=40**，且前 **20** epoch 不允许早停（避免验证噪声一上来就误杀）。  
- 实际：**第 24 轮**达到验证集上的最佳 checkpoint；之后验证指标长期没有再超过这一轮，一直拖到 **第 64 轮**触发早停——也就是说 **模型「很早」就达到了后来被保存下来的权重**，后面主要是在小幅震荡里耗时间。

---

## 3. 关键超参数一览（便于复现）

| 类别 | 取值 |
|------|------|
| seed | 28 |
| batch_size | 32 |
| learning_rate | 1e-4 |
| lr_scheduler | plateau（ReduceLROnPlateau） |
| lr_scheduler_patience | 5（args 快照）；Plateau 监控验证 loss |
| scheduler_min_lr | 1e-6 |
| early_stop_patience / min_epochs | 40 / 20 |
| select_best | **balanced_acc** |
| glevel_arch | shared_mlp |
| fused_layer_norm | 是 |
| cross_modal_attn / layers | 是 / 1 |
| mlp_dropout | 0.25 |
| modality_dropout_p | 0.12 |
| label_smoothing | 0.05 |
| sampler_medium_boost | 1.5 |
| mixup_prob | 0 |
| 特征维度 audio / video / text | 512 / 512 / 2560 |

更细的字段以 **`logs/args_glevel_20260509_150130.json`** 为准。

---

## 4. 数字总览：训练集 / 验证集 / 官方测试一眼对比

下面这张表把「同一套权重」在不同数据上的行为放在一起，避免只看验证忘了训练、或只看训练忘了榜单。

### 4.1 总表（强烈建议先看这一行）

| 划分 | 样本数 | 准确率 acc | 均衡准确率 bal_acc | 宏平均 F1 | 备注 |
|------|--------|------------|---------------------|-----------|------|
| **训练集** | **450** | **0.9867** | **0.9871** | **0.9867** | 见 §4.2：如何得到 |
| **验证集** | **63**（CSV 64 行剔除 1 条缺特征） | **0.6032** | **0.5861** | **0.5846** | 与 `train.log` / 后验 eval 一致 |
| **官方测试集** | **130** | **0.46923** | — | — | 平台返回；本地无标签 |

**一眼能读出的故事**：

- 在 **训练集**上，模型几乎能把样本「背」得非常好（acc **≈98.7%**）。  
- 换到 **验证集**，acc 掉到 **≈60%** —— 中间差了 **约 38 个百分点**。  
- 再换到 **官方测试**，acc **≈46.9%** —— 又比验证 **低约 13 个百分点**。

这不是排版错误，而是典型的 **「训练拟合极强 →  Hold-out 泛化台阶式下降」** 轮廓；后面 §6 会逐项拆开可能原因。

### 4.2 训练集准确率从哪里来（避免和 epoch 日志混淆）

训练过程中每个 epoch 的日志里只打印 **`train_ce`（训练交叉熵）**，**没有**打印 `train_acc`。为了在报告里给出 **可比的准确率**，在 **`best.pth` 固定不变** 的前提下，用仓库脚本对 **整张 `train_data.csv`** 做了一次 **eval 模式、按样本顺序前向**：

```bash
# 工程根目录；Python 使用与本仓库一致的隔离环境（示例：.venv_glevel_cpu）
python tools/eval_glevel_checkpoint_on_csv.py \
  --eval_csv /data/Super-Lu/dataset/train_data.csv \
  --rating_csv /data/Super-Lu/dataset/train_data.csv \
  --labels_in_split_csv --g_level_int_encoding one \
  --train_audio_dir /data/Super-Lu/dataset/train_feature/audio \
  --train_video_dir /data/Super-Lu/dataset/train_feature/video \
  --train_text_dir ./data/text_nb \
  --checkpoint experiments/gpu_combo_sweep/mass_20260509_145045/S_plateau_ln/seed28/best.pth \
  --text_dim 2560 --cross_modal_attn --cross_modal_layers 1 --fused_layer_norm \
  --num_workers 0 --batch_size 32
```

得到的 **`[metrics_line_local]`** 为：`acc=0.9867`，`bal_acc=0.9871`，`macro_f1=0.9867`，`n=450`。  
这与 **epoch 24** 时日志里的 **`train_ce≈0.256`** 同时成立：CE 已经很低，全量扫一遍训练集时多数样本预测正确 —— **两条证据指向同一结论：训练拟合程度很高。**

同一脚本对 **验证集**再跑一遍，得到 **`acc=0.6032`**，与训练日志 `val_summary` **完全一致**，说明评估口径对齐。

### 4.3 验证集：除了准确率，模型错在哪

验证集中 **Low / Medium / High** 的真实条数约为 **21 / 17 / 25**（来自 `train.log` 开头的计数）。最佳 checkpoint 上的 **混淆矩阵（行=真值，列=预测）** 为：

| 真 \\ 预测 | Low | Medium | High |
|------------|-----|--------|------|
| **Low** | 14 | 4 | 3 |
| **Medium** | 5 | 7 | 5 |
| **High** | 4 | 4 | 17 |

 sklearn 分类报告摘要（support = 该类真实条数）：

| 类 | precision | recall | F1 |
|----|-----------|--------|-----|
| Low | 0.609 | 0.667 | 0.636 |
| Medium | 0.467 | 0.412 | 0.438 |
| High | 0.680 | 0.680 | 0.680 |

**直观结论**：模型在 **High** 上最稳；**Medium** 最容易被当成 Low 或 High（边界样本多），这也和主观等级连续、标注噪声的典型情况一致。

验证阶段日志还提到：**错分 25/63**；全体样本「top2 置信度间隔」均值约 **0.611**，错分子集间隔反而略高（约 **0.671**）—— 说明不少错误并不是「模型完全糊掉」，而是 **在相近分数之间选错边**，迁移到更难分布的测试集时更容易翻车。

### 4.4 测试集：官方分数与本地能算的侧面指标

- **官方测试准确率（已确认）**：**`0.46923`**（约 **46.92%**；130 条里约合 **61** 条对）。  
- **本地只能看到预测文件**：该提交的测试预测类别计数约为 **45 / 36 / 49**（等级 1/2/3），占比约 **34.6% / 27.7% / 37.7%**。  
- **`reports/deepseek_zyn_comparison_table.csv`** 里与参考列 **`DeepSeek_zyn_majority_vote`** 的逐样本一致率约 **30.8%** —— 只能说明「和某条参考流水线不像」，**不能换算成官方准确率**。  
- **`heuristic_test_recovery/example_manifest.json`** 里把本提交写成 **`accuracy":"0.46923"`**，并把几条同伴示例写在 **约 0.5077～0.5539**：用来驱动启发式恢复脚本，**不是**赛方完整排行榜，但能侧面说明 **在同一张示例表里本提交的测试分低于那些条目**。

---

## 5. 和其它结果比：到底「赢在哪、输在哪」

### 5.1 先把尺子说清楚：大家在比什么

- **训练集 / 验证集**：同一套赛方划分时，**验证集数字可以直接横向比**（前提是同一剔除规则、同一特征版本）。  
- **官方测试**：只有平台分数可比；**验证高不代表测试高**——本模型就是教科书级反例。

### 5.2 同一轮 masswinner 刷新（iter2）：验证集上谁高

下面四条都来自 **`mass_winners_refresh_iter2_summary.tsv`**，**验证口径一致**（同一 val CSV、同一套特征）。数值摘自仓库摘要。

| 提交（peer 文件名关键词） | val_acc | val_bal_acc | val_macro_f1 | val_ce |
|---------------------------|---------|-------------|--------------|--------|
| **masswinner_S_plateau_ln_seed28（本模型）** | **0.6032** | **0.5861** | **0.5846** | 1.5867 |
| masswinner_S_ref_plateau_seed37 | 0.5873 | 0.5414 | 0.5079 | 1.8712 |
| masswinner_S_ref_cosine_seed5 | 0.5873 | 0.5691 | 0.5667 | 1.3860 |
| masswinner_S_ref_plateau_seed10 | 0.5714 | 0.5520 | 0.5429 | 1.5884 |
| masswinner_S_ref_cosine_seed10 | 0.5714 | 0.5444 | 0.5383 | 1.2389 |
| masswinner_S_ref_plateau_seed99 | 0.5556 | 0.5297 | 0.5286 | 1.6617 |

**一句话**：在「这一轮挑出来对外刷新文件」的候选里，**本模型在验证集准确率与均衡准确率上都更靠前**，所以被选成 masswinner 路线里的主力并不奇怪。

### 5.3 官方测试：同一示例 manifest 里的差距（便于直觉对齐）

| 条目（示例 manifest） | 登记的测试准确率（小数） | 约等于百分数 |
|----------------------|---------------------------|--------------|
| 同伴示例 submission2 | 0.50769 | ~50.8% |
| 同伴示例 submission_ | 0.53846 | ~53.8% |
| 同伴示例 submission5 | 0.55385 | ~55.4% |
| **本提交 masswinner_S_plateau_ln_seed28** | **0.46923** | **~46.9%** |

也就是说：**在验证集上领先的一批模型里，这一条在官方测试上落在了示例同伴分数带的下方。** 这不矛盾，只能说明 **验证集太小 / 与测试分布不一致 / 或挑选偏差** 等因素在起作用（详见 §6）。

### 5.4 sweep 内部多目标排序（`ranking_triple.tsv`）

在 **`mass_20260509_145045/ranking_triple.tsv`** 里，每个候选除了 **val_acc**，还有与加权伪标签一致性、与 DeepSeek 参考一致性等列，并合成 **`composite`**。  
本模型一行约为：**val_acc=0.6032**，`weighted_pseudo_agree≈0.557`，`deepseek_agree≈0.362`。  
**通俗讲**：它在「验证分数」上很漂亮，但「和某些外部参考预测的逐样本重合度」并不是全场最高——说明 **不同目标会挑出不同模型**，若你的目标是刷榜单测试，不宜只看 val 一条曲线。

---

## 6. 为什么验证很高、官方测试只有约 0.47？（结合 §4.1 的数字一起读）

下面每一条都可以单独造成落差；实战中往往是 **多条叠加**。

1. **验证集太小（63 条），随机波动大**  
   多对一条、少对一条，准确率台阶大约是 **1/63≈1.6%**。换句话说，**验证集上的「60%」统计上并没有想象中稳**；测试集 **130 条**反而更「钝」、更能反映稳定误差。

2. **训练集几乎拟合满了（acc≈98.7%）**  
   与 **val≈60%** 的巨大鸿沟说明：模型 **很强地记住了训练分布**；一旦测试分布偏离训练分布，分数往下掉非常正常。

3. **选模指标是 balanced_acc，不是测试 acc**  
   保存 checkpoint 时优先三类均衡表现；赛方测试通常是 **整体准确率**。二者最优解 **不必一致**。

4. **大规模网格 + 多 seed：隐性「多次试探验证集」**  
   从许多 (combo, seed) 里挑出验证最好的一条，会把验证分数 **系统性抬高**；测试集从未参与选择，自然 **不会享受这份乐观**。

5. **分布偏移 / 领域gap**  
   测试受试人群、录制环境、题目难度、特征提取版本任一不一致，都会造成 **同一权重** 在测试上整体变差；多模态模型还可能 **过度依赖某一模态在训练上的捷径**，测试上该模态信号一变就崩。

6. **特征与 Nanbeige 文本链路一致性**  
   train / val / test 三路目录是否同源、是否有静默缺失或填充差异，都会吃掉几个百分点。

7. **类别先验不匹配**  
   训练时用采样器「微调」了 Medium；若真实测试标签分布不同，**整体 acc** 会对决策边界非常敏感。

8. **标注噪声与等级边界**  
   Medium 与相邻等级混淆（§4.3）在验证上已经可见；测试若标注更严或标准略有不同，会放大这种现象。

9. **高分同伴模型可能在集成、校准或后处理上不同**  
   仅凭单一 shared_mlp checkpoint，即使验证赢了同一批 masswinner，也可能在榜单上输给 **结构不同或多数模型投票** 的方案。

**建议的下一步（仍然直观）**：若要做反驳实验，优先试 **更强正则 / 早停更早截在 train-val 间隙更小处 / 测试时增强 / 多模型集成 / 目标贴近测试（需可靠 proxy）**，再看官方测试是否脱离 **0.47** 平台。

---

## 7. 综合结论

1. **模型画像**：全模态 Shared MLP + **LayerNorm** + **跨模态注意力** + **Plateau 学习率** + **Nanbeige 文本**，用 **验证均衡准确率** 选 checkpoint；在 **mass_20260509_145045** 中 **`S_plateau_ln` / seed 28** 的表现使其进入 masswinner 主线。  
2. **数据事实**：**训练 acc≈98.7%**，**验证 acc≈60.3%**，**官方测试 acc≈46.9%** —— 三层落差清晰，不能把验证分数当成测试承诺。  
3. **横向对比**：**验证集上**优于同期多条 masswinner；**示例 manifest 登记的测试准确率上**低于几条同伴条目。  
4. **复现入口**：`train_task2_glevel.py`；组合 **`S_plateau_ln`** 定义于 `tools/run_glevel_gpu_combo_sweep.sh`。

---

## 8. 报告依据与复算命令

- **训练与验证日志**：`experiments/gpu_combo_sweep/mass_20260509_145045/S_plateau_ln/seed28/train.log`  
- **训练参数**：`logs/args_glevel_20260509_150130.json`  
- **训练集 / 验证集准确率（后验 eval）**：`tools/eval_glevel_checkpoint_on_csv.py`，命令见 **§4.2**（评测使用 `.venv_glevel_cpu/bin/python` 可避免部分系统上 sklearn/pyarrow 冲突）。  
- **masswinner 摘要**：`experiments/glevel_improvement_plan/mass_winners_refresh_iter2_summary.tsv`  
- **对比表**：`reports/deepseek_zyn_comparison_table.csv`、`experiments/gpu_combo_sweep/mass_20260509_145045/ranking_triple.tsv`  
- **官方测试准确率**：**0.46923**（用户 / 平台确认）

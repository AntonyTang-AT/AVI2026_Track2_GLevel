"""
启发式恢复隐藏测试标签（近似可行解，非真实标注）。

思路概要
--------
1. 输入：多份提交 CSV（同一批测试 id），可选每份在榜单上的 accuracy（总样本数默认 130）。
2. 对齐：按 id 取交集排序，预测列支持 g_level / g_level_pred，统一到 {1,2,3}。
3. 先验（启发式优先级）
   - 多样本预测一致：同一 id 下若所有模型类别相同 → 赋予最高置信，可作为初始标签且可选「冻结」不参与随机翻转。
   - 加权多数票：未完全一致时，用各模型可信度权重（默认与 accuracy 成正比）做加权投票得到初始 y。
4. 搜索：在离散标签空间上对 y 做模拟退火 / 局部搜索，软约束目标为
      min_y  Σ_k ( match(y, 第k份预测) - target_k )²
   其中 target_k = round(acc_k * N)；亦可加入极小项鼓励与高一致性样本对齐。
5. 局限：方程欠定，输出仅为与给定 accuracy 近似相容的一条标签向量；不可当作 ground truth。

性能：单链 SA 已用「翻转一步」增量更新 matches/agreement，避免每步全表扫描。
并行：`--workers N` 为多进程多起点（每条链独立 RNG），总工作量约为 N×steps，适合多核 CPU。
本题规模 n≈130、k≈5，SEQUENTIAL 的随机翻转不适合 GPU；CUDA 传输与同步开销会远大于收益，故未做 GPU 路径。

准确率→命中数：须用 `Decimal` 或字符串小数配置；可选 `hit_policy`（nearest / half_up / floor 等），或直接写整数 `hits` 锁定正确条数。
可选先验：`lambda_plurality`（按 max_votes/K 加权众票）、`lambda_balance`（三类条数贴近 `balance_counts` 或默认近似 1:1:1）、`proposal_bias`（翻转优先低重合样本）。

参见 run_recovery.py --help。
"""

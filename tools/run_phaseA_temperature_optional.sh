#!/usr/bin/env bash
# Phase A 可选：标量温度 T（CE）。步骤：
#  1) only_test / 训练结束 时设置 VAL_ERRORS_CSV，确保写出含 logits 的报告（若当前 train_task2 仅写 prob，请改用 eval 脚本导出 logits npz）。
#  2) 准备 npz：键 logits (N,K)、labels (N,)
#  3) python tools/fit_temperature_scaling.py --probs_npz ... --out_json experiments/glevel_improvement_plan/temperature_T.json
#  4) only_test 增加 --calib_temperature_json 指向上面的 JSON
#
# 若尚无 logits npz，可跳过本步骤；本轮计划以 infer_logit_bias 细扫为主。
echo "[phaseA_temperature_optional] 见脚本内注释；无 logits npz 时跳过。"
exit 0

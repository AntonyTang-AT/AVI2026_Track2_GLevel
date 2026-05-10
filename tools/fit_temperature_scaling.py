#!/usr/bin/env python3
"""
对 eval_glevel_checkpoint_on_csv.py --dump_probs 的 CE logits 拟合标量温度 T（最小化 NLL）。
输出 JSON：{"T": float}，供 train_task2_glevel --calib_temperature_json 使用。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs_npz", required=True, help="含 logits (N,K) 与 labels (N,)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--min_T", type=float, default=0.05)
    ap.add_argument("--max_T", type=float, default=10.0)
    args = ap.parse_args()

    z = np.load(args.probs_npz)
    logits = torch.from_numpy(z["logits"].astype(np.float32))
    labels = torch.from_numpy(z["labels"].astype(np.int64))

    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.25, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = T.clamp(min=args.min_T, max=args.max_T)
        loss = F.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    opt.step(closure)
    t_final = float(T.clamp(min=args.min_T, max=args.max_T).detach().item())
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"T": t_final}, f, indent=2)
    with torch.no_grad():
        nll_before = F.cross_entropy(logits, labels).item()
        nll_after = F.cross_entropy(logits / t_final, labels).item()
    print(
        f"[fit_temperature_scaling] T={t_final:.6f} nll_before={nll_before:.4f} nll_after={nll_after:.4f} → {args.out_json}",
        flush=True,
    )
    print(
        f"[metrics_line_local] method=temperature T={t_final:.6f} nll_after={nll_after:.6f} out={args.out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()

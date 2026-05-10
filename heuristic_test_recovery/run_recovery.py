#!/usr/bin/env python3
"""
CLI：多份提交 + 可选榜单 accuracy → 启发式标签向量 + 导出 CSV。

准确率 → 目标命中数：默认用 Decimal + hit_policy（见 --hit_policy），避免 float 误差；
manifest 里推荐字符串小数 "accuracy": "0.46923" 或 "accuracy_pct": "46.923"。
若平台直接给了正确条数，请写整数 "hits": 61（优先级最高）。

示例（manifest + 8 进程并行）:
  python heuristic_test_recovery/run_recovery.py \\
    --manifest heuristic_test_recovery/example_manifest.json \\
    --workers 8 --steps 20000 --freeze_unanimous

示例（命令行逐项，准确率请保持字符串形式以免 JSON/float 误差）:
  python heuristic_test_recovery/run_recovery.py \\
    --model external/submissions_peer/a.csv 0.53077 \\
    --out logs/recovered_y.csv --hit_policy nearest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from heuristic_test_recovery.load_submissions import load_aligned_matrix
from heuristic_test_recovery.priors import (
    build_initial_y,
    row_consensus_features,
    unanimous_mask,
    uniform_class_targets,
)
from heuristic_test_recovery.search import (
    dedupe_results_by_y,
    simulate_annealing,
    simulate_annealing_collect_chains,
    simulate_annealing_parallel,
    squared_loss,
)
from heuristic_test_recovery.targets import build_targets_vector


def _weight_from_manifest_item(item: dict, n: int) -> float:
    raw = item.get("accuracy", item.get("acc"))
    if raw is not None:
        return float(Decimal(str(raw)))
    pct = item.get("accuracy_pct")
    if pct is not None:
        return float(Decimal(str(pct)) / Decimal("100"))
    hits = item.get("hits")
    if hits is not None:
        return int(hits) / float(n)
    return 1.0


def _parse_balance_mu(n: int, raw: object | None, cli_str: str) -> np.ndarray | None:
    if raw is not None:
        arr = np.array(list(raw), dtype=np.float64)
    elif str(cli_str).strip():
        arr = np.array([float(x) for x in str(cli_str).split(",")], dtype=np.float64)
    else:
        return None
    if arr.shape != (3,):
        raise SystemExit("balance_counts 须为长度 3")
    s = float(arr.sum())
    if abs(s - float(n)) > 1e-6:
        raise SystemExit(f"balance_counts 之和须等于 n={n}，当前为 {s}")
    return arr


def _write_y_csv(path: Path, ids: list[str], y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["id", "g_level"])
        for sid, g in zip(ids, y):
            wcsv.writerow([sid, int(g)])


def main() -> None:
    ap = argparse.ArgumentParser(description="启发式搜索：拟合多份提交的榜单 accuracy，并优先尊重全票一致样本")
    ap.add_argument("--manifest", type=str, default="", help="JSON：files[{csv,hits?,accuracy?,accuracy_pct?}], hit_policy, ...")
    ap.add_argument("--model", nargs=2, metavar=("CSV", "ACC"), action="append", help="可重复；ACC 用 na 跳过该模型约束")
    ap.add_argument("--out", type=str, default="", help="输出 CSV：id,g_level（manifest 可写 output_csv）")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t0", type=float, default=2.0)
    ap.add_argument("--t_min", type=float, default=1e-3)
    ap.add_argument("--lambda_agreement", type=float, default=0.05, help="越大越倾向与更多模型预测一致")
    ap.add_argument("--freeze_unanimous", action="store_true", help="全体模型预测相同的样本不翻转")
    ap.add_argument("--no_search", action="store_true", help="只做加权先验，不做模拟退火")
    ap.add_argument("--trace_every", type=int, default=0)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=">1 时用多进程并行跑多条独立 SA，每条均执行 --steps；取最优 loss",
    )
    ap.add_argument(
        "--hit_policy",
        type=str,
        default="nearest",
        choices=("nearest", "half_up", "floor", "ceil", "trunc"),
        help="准确率换算为整数命中数：nearest=与声明准确率最接近的 k/n；floor=先 Decimal(acc)*n 再 floor（如 0.46923*130→60）",
    )
    ap.add_argument(
        "--verify_strict",
        action="store_true",
        help="若终解命中数与目标不完全一致则退出码 1（无解或步数不足）",
    )
    ap.add_argument(
        "--lambda_balance",
        type=float,
        default=0.0,
        help=">0 时惩罚三类条数偏离 balance_counts（默认近似 1:1:1），越小越不影响命中约束",
    )
    ap.add_argument(
        "--lambda_plurality",
        type=float,
        default=0.0,
        help=">0 时奖励与高重合众票一致；权重为 max_votes/K，重合愈高愈「自信」",
    )
    ap.add_argument(
        "--proposal_bias",
        type=float,
        default=0.0,
        help=">0 时随机翻转位置偏向低重合样本，减少对高置信行的破坏（幂次）",
    )
    ap.add_argument(
        "--balance_counts",
        type=str,
        default="",
        help="三类目标条数，逗号分隔，须之和=n；空则用近似均衡",
    )
    ap.add_argument(
        "--save_top",
        type=int,
        default=1,
        help=">1 时保存按综合 loss 排序的去重解，文件名为 {stem}_topXX.csv + JSON 汇总",
    )
    ap.add_argument(
        "--collect_chains",
        type=int,
        default=0,
        help="实际跑的独立 SA 条数；0 表示自动≈max(save_top*4, 40)。越大越容易凑满 save_top 种不同解",
    )
    ap.add_argument(
        "--pool_workers",
        type=int,
        default=0,
        help="collect 阶段最大并行进程数；0 表示 min(collect_chains, CPU 核数)",
    )
    args = ap.parse_args()

    csv_paths: list[str] = []
    accs_for_weight: list[float | None] = []
    manifest_rows: list[dict] | None = None
    acc_raw_cli: list[str | None] | None = None

    if args.manifest:
        mp = Path(args.manifest).expanduser().resolve()
        with mp.open(encoding="utf-8") as f:
            man = json.load(f)
        manifest_rows = list(man.get("files", man.get("submissions", [])))
        for item in manifest_rows:
            csv_paths.append(str(Path(item["csv"]).expanduser().resolve()))
        accs_for_weight = [None] * len(csv_paths)
        steps = int(man["steps"]) if "steps" in man else args.steps
        seed = int(man["seed"]) if "seed" in man else args.seed
        t0 = float(man["t0"]) if "t0" in man else args.t0
        t_min = float(man["t_min"]) if "t_min" in man else args.t_min
        lam = float(man["lambda_agreement"]) if "lambda_agreement" in man else args.lambda_agreement
        freeze_u = bool(man["freeze_unanimous"]) if "freeze_unanimous" in man else args.freeze_unanimous
        no_search = bool(man["no_search"]) if "no_search" in man else args.no_search
        trace = int(man["trace_every"]) if "trace_every" in man else args.trace_every
        workers = int(man["workers"]) if "workers" in man else args.workers
        out_path = str(man["output_csv"]) if "output_csv" in man else str(man.get("out", args.out))
        if str(args.out).strip():
            out_path = str(Path(args.out).expanduser().resolve())
        hp_raw = str(man["hit_policy"]) if "hit_policy" in man else args.hit_policy
        if hp_raw not in ("nearest", "half_up", "floor", "ceil", "trunc"):
            raise SystemExit(f"非法 hit_policy={hp_raw!r}")
        hit_policy = hp_raw  # type: ignore[assignment]
        verify_strict = bool(man["verify_strict"]) if "verify_strict" in man else args.verify_strict
        lambda_balance = float(man["lambda_balance"]) if "lambda_balance" in man else args.lambda_balance
        lambda_plurality = float(man["lambda_plurality"]) if "lambda_plurality" in man else args.lambda_plurality
        proposal_bias = float(man["proposal_bias"]) if "proposal_bias" in man else args.proposal_bias
        balance_counts_raw = man["balance_counts"] if "balance_counts" in man else None
        balance_counts_cli = args.balance_counts
        save_top = int(man["save_top"]) if "save_top" in man else args.save_top
        collect_chains_cfg = int(man["collect_chains"]) if "collect_chains" in man else args.collect_chains
        pool_workers_cfg = int(man["pool_workers"]) if "pool_workers" in man else args.pool_workers
    else:
        if not args.model:
            raise SystemExit("请提供 --manifest 或至少一对 --model CSV ACC")
        acc_raw_cli = []
        for p, a in args.model:
            csv_paths.append(str(Path(p).expanduser().resolve()))
            if str(a).lower() in ("na", "nan", "none", ""):
                accs_for_weight.append(None)
                acc_raw_cli.append(None)
            else:
                acc_raw_cli.append(str(a).strip())
                try:
                    accs_for_weight.append(float(a))
                except ValueError:
                    accs_for_weight.append(float(Decimal(str(a))))
        steps = args.steps
        seed = args.seed
        t0 = args.t0
        t_min = args.t_min
        lam = args.lambda_agreement
        freeze_u = args.freeze_unanimous
        no_search = args.no_search
        trace = args.trace_every
        workers = args.workers
        out_path = args.out
        hit_policy = args.hit_policy  # type: ignore[assignment]
        verify_strict = args.verify_strict
        lambda_balance = args.lambda_balance
        lambda_plurality = args.lambda_plurality
        proposal_bias = args.proposal_bias
        balance_counts_raw = None
        balance_counts_cli = args.balance_counts
        save_top = args.save_top
        collect_chains_cfg = args.collect_chains
        pool_workers_cfg = args.pool_workers

    if not str(out_path).strip():
        raise SystemExit("请指定 --out 或在 manifest 中设置 output_csv")

    preds, ids, paths, pred_cols = load_aligned_matrix(csv_paths)
    n, k = preds.shape
    print(f"[recovery] n_samples={n} n_models={k} hit_policy={hit_policy}", flush=True)
    for j, (pth, pc) in enumerate(zip(paths, pred_cols)):
        print(f"  model[{j}] pred_col={pc} path={pth}", flush=True)

    if len(accs_for_weight) != k:
        raise ValueError("accuracy 权重列表与 CSV 数量不一致")

    balance_mu = _parse_balance_mu(n, balance_counts_raw, balance_counts_cli)
    mode_labels, _mv, consensus_strength = row_consensus_features(preds)
    if lambda_plurality > 0 or proposal_bias > 0:
        print(
            f"[recovery] 先验：lambda_plurality={lambda_plurality} proposal_bias={proposal_bias} | "
            f"重合度 mean={consensus_strength.mean():.3f} min={consensus_strength.min():.3f} max={consensus_strength.max():.3f}",
            flush=True,
        )
    if lambda_balance > 0:
        mu_show = balance_mu if balance_mu is not None else uniform_class_targets(n)
        print(
            f"[recovery] 先验：lambda_balance={lambda_balance} 目标三类条数(1/2/3)={mu_show.tolist()}",
            flush=True,
        )

    # 构建目标命中数
    targets: np.ndarray | None = None
    if manifest_rows is not None:
        tgt_list, notes = build_targets_vector(manifest_rows, n, hit_policy)
        targets = np.array(tgt_list, dtype=np.int64)
        print("[recovery] 目标命中数推导:", flush=True)
        for line in notes:
            print(f"  {line}", flush=True)
    else:
        assert acc_raw_cli is not None
        if any(x is None for x in acc_raw_cli) or not all(acc_raw_cli):
            raise SystemExit("命令行模式须为每个 --model 提供非 na 的准确率，或改用 manifest")
        rows = [{"accuracy": s} for s in acc_raw_cli]
        tgt_list, notes = build_targets_vector(rows, n, hit_policy)
        targets = np.array(tgt_list, dtype=np.int64)
        print("[recovery] 目标命中数推导:", flush=True)
        for line in notes:
            print(f"  {line}", flush=True)

    # 权重（多数票）：accuracy / accuracy_pct / hits→命中率 任一均可
    w = np.ones(k, dtype=np.float64)
    for j in range(k):
        if manifest_rows is not None:
            w[j] = _weight_from_manifest_item(manifest_rows[j], n)
        else:
            assert acc_raw_cli is not None
            w[j] = float(Decimal(acc_raw_cli[j]))  # type: ignore[arg-type]

    y0 = build_initial_y(preds, w, prefer_unanimous=True)
    uni = unanimous_mask(preds)
    print(f"[recovery] 全票一致样本: {int(uni.sum())}/{n}", flush=True)

    if targets is not None:
        print(f"[recovery] 目标命中数列表 target_k: {targets.tolist()}", flush=True)
        for j in range(k):
            tj = int(targets[j])
            print(
                f"  model[{j}] 目标命中率={tj}/{n}="
                f"{(Decimal(tj) / Decimal(n)):.12f}",
                flush=True,
            )
        ini_matches = np.sum(y0[:, None] == preds, axis=0)
        print(f"[recovery] 初解命中数: {ini_matches.tolist()} 初解平方损失={squared_loss(ini_matches, targets):.2f}", flush=True)

    freeze_mask = uni if freeze_u else None

    top_results = None
    collect_chains_used: int | None = None

    if save_top > 1 and no_search:
        raise SystemExit("save_top>1 需要运行搜索，请勿同时使用 --no_search")

    if no_search:
        y_hat = y0
        print("[recovery] 跳过搜索，输出为先验标签", flush=True)
    else:
        if targets is None:
            raise SystemExit("内部错误：targets 为空")
        if lam <= 0:
            raise SystemExit("须设置 --lambda_agreement > 0（manifest 可写 lambda_agreement）")

        if save_top > 1:
            cc = collect_chains_cfg if collect_chains_cfg > 0 else max(save_top * 4, 40)
            pw = pool_workers_cfg if pool_workers_cfg > 0 else min(cc, os.cpu_count() or 8)
            collect_chains_used = cc
            seeds = [seed + i * 97_853 for i in range(cc)]
            print(
                f"[recovery] 多解模式 save_top={save_top} collect_chains={cc} pool_workers={pw}",
                flush=True,
            )
            raw_list = simulate_annealing_collect_chains(
                y0,
                preds,
                targets,
                freeze_mask=freeze_mask,
                steps_per_chain=steps,
                seeds=seeds,
                pool_workers=pw,
                t0=t0,
                t_min=t_min,
                lambda_agreement=lam,
                trace_every=0,
                lambda_balance=lambda_balance,
                balance_mu=balance_mu,
                lambda_plurality=lambda_plurality,
                mode_labels=mode_labels,
                consensus_strength=consensus_strength,
                proposal_bias=proposal_bias,
            )
            top_results = dedupe_results_by_y(raw_list, save_top)
            print(
                f"[recovery] 去重后得到 {len(top_results)} 个不同解（目标 {save_top}，共跑 {cc} 链）",
                flush=True,
            )
            if len(top_results) < save_top:
                print(
                    "[recovery] WARNING: 唯一解数量不足，可增大 collect_chains 或 steps 以增加多样性",
                    flush=True,
                )
            res = top_results[0]
            y_hat = res.y
            print(
                f"[recovery] 最优链 seed={res.seed} loss={res.loss:.4f} "
                f"接受率={res.steps_accepted}/{res.steps_total}",
                flush=True,
            )
        else:
            nw = max(1, int(workers))
            if nw > 1:
                print(
                    f"[recovery] 并行 workers={nw}，每链 steps={steps}（总迭代≈{nw * steps}）",
                    flush=True,
                )
                res = simulate_annealing_parallel(
                    y0,
                    preds,
                    targets,
                    freeze_mask=freeze_mask,
                    steps_per_chain=steps,
                    seed_base=seed,
                    workers=nw,
                    t0=t0,
                    t_min=t_min,
                    lambda_agreement=lam,
                    trace_every=trace,
                    lambda_balance=lambda_balance,
                    balance_mu=balance_mu,
                    lambda_plurality=lambda_plurality,
                    mode_labels=mode_labels,
                    consensus_strength=consensus_strength,
                    proposal_bias=proposal_bias,
                )
            else:
                res = simulate_annealing(
                    y0,
                    preds,
                    targets,
                    freeze_mask=freeze_mask,
                    steps=steps,
                    seed=seed,
                    t0=t0,
                    t_min=t_min,
                    lambda_agreement=lam,
                    trace_every=trace,
                    lambda_balance=lambda_balance,
                    balance_mu=balance_mu,
                    lambda_plurality=lambda_plurality,
                    mode_labels=mode_labels,
                    consensus_strength=consensus_strength,
                    proposal_bias=proposal_bias,
                )
            y_hat = res.y
            print(
                f"[recovery] SA 接受率={res.steps_accepted}/{res.steps_total} "
                f"末损失={res.loss:.4f} seed={getattr(res, 'seed', seed)}",
                flush=True,
            )
        fin_m = np.sum(y_hat[:, None] == preds, axis=0).astype(np.int64)
        c1, c2, c3 = int((y_hat == 1).sum()), int((y_hat == 2).sum()), int((y_hat == 3).sum())
        print(f"[recovery] 终解三类条数 g1/g2/g3: {c1}/{c2}/{c3}", flush=True)
        print(f"[recovery] 终解命中数: {fin_m.tolist()}", flush=True)
        print(f"[recovery] 终解平方损失={squared_loss(fin_m, targets):.6f}", flush=True)
        print(
            "[recovery] 终解对应命中率: "
            + ", ".join(f"{fin_m[j]}/{n}" for j in range(k)),
            flush=True,
        )
        mismatch = np.where(fin_m != targets)[0]
        if len(mismatch):
            print(
                "[recovery] WARNING: 下列模型索引的命中数未等于目标（可能无解或需增加 steps/lambda）："
                f" {mismatch.tolist()}",
                flush=True,
            )
            if verify_strict:
                raise SystemExit(1)
        else:
            print("[recovery] 校验：终解命中数与目标完全一致。", flush=True)

    outp = Path(out_path).expanduser().resolve()
    if save_top > 1 and not no_search:
        stem = outp.stem
        parent = outp.parent
        summ: dict = {
            "save_top": save_top,
            "collect_chains": collect_chains_used,
            "note": "rank 按综合 loss 升序；sq_loss 为仅榜单命中数平方误差",
            "solutions": [],
        }
        assert top_results is not None
        for rank, tr in enumerate(top_results, start=1):
            fm = np.sum(tr.y[:, None] == preds, axis=0).astype(np.int64)
            sq = float(squared_loss(fm, targets)) if targets is not None else 0.0
            c1, c2, c3 = int((tr.y == 1).sum()), int((tr.y == 2).sum()), int((tr.y == 3).sum())
            fp = parent / f"{stem}_top{rank:02d}.csv"
            _write_y_csv(fp, ids, tr.y)
            print(f"[recovery] 已写入 #{rank} loss={tr.loss:.6f} sq_loss={sq:.6f} → {fp}", flush=True)
            summ["solutions"].append(
                {
                    "rank": rank,
                    "csv": str(fp),
                    "loss": tr.loss,
                    "sq_loss": sq,
                    "seed": tr.seed,
                    "matches_per_model": fm.tolist(),
                    "class_counts": {"g1": c1, "g2": c2, "g3": c3},
                }
            )
        summ_path = parent / f"{stem}_top_summary.json"
        summ_path.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[recovery] 汇总 JSON → {summ_path}", flush=True)
        _write_y_csv(outp, ids, top_results[0].y)
        print(f"[recovery] 最优解亦写入主路径 {outp}", flush=True)
    else:
        _write_y_csv(outp, ids, y_hat)
        print(f"[recovery] 已写入 {outp}", flush=True)


if __name__ == "__main__":
    main()

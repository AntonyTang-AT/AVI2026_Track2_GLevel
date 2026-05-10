#!/usr/bin/env python3
"""
在 prepare_svm_data 生成的 .npy 上训练传统 ML 基线，结果追加到 logs/svm_results.txt。
- StandardScaler（训练集 fit）
- 可选 PCA(n_components)
- RBF-SVM：GridSearchCV（f1_macro，内层 5 折），在完整训练集 refit
- 在 hold-out X_val/y_val 上报告 acc / macro-F1 / per-class recall / 混淆矩阵
- LinearSVC、RandomForest；XGBoost 若未安装则跳过并记日志
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _append_log(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _val_report(name: str, y_va: np.ndarray, pred: np.ndarray) -> str:
    acc = accuracy_score(y_va, pred)
    mf1 = f1_score(y_va, pred, average="macro", zero_division=0)
    rec = recall_score(y_va, pred, average=None, zero_division=0)
    cm = confusion_matrix(y_va, pred)
    lines = [
        f"\n=== {name} ===",
        f"val_acc={acc:.4f} val_macro_f1={mf1:.4f}",
        f"per_class_recall={rec.tolist()}",
        f"confusion_matrix:\n{cm}",
        classification_report(y_va, pred, zero_division=0),
    ]
    return "\n".join(lines)


def _fit_report(name: str, clf, X_tr, y_tr, X_va, y_va) -> str:
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    return _val_report(name, y_va, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default=os.path.join(_ROOT, "data", "svm"),
        help="含 X_train.npy 等（默认 data/svm）",
    )
    ap.add_argument("--log_file", default=os.path.join(_ROOT, "logs", "svm_results.txt"))
    ap.add_argument("--pca", type=int, default=0, help="PCA 维数；0 关闭")
    ap.add_argument(
        "--rbf_cv",
        type=int,
        default=5,
        help="RBF-SVM GridSearch 折数（计划默认 5）",
    )
    ap.add_argument(
        "--save_sklearn_bundle",
        default="",
        help="可选：将最佳 RBF Pipeline 保存为 joblib（供 predict_submission --backend sklearn）",
    )
    args = ap.parse_args()

    ts = datetime.now().isoformat(timespec="seconds")
    header = f"\n\n######## train_svm {ts} ########\ndata_dir={args.data_dir}\n"
    _append_log(args.log_file, header)
    print(header, flush=True)

    X_tr = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_tr = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_va = np.load(os.path.join(args.data_dir, "X_val.npy"))
    y_va = np.load(os.path.join(args.data_dir, "y_val.npy"))

    out_lines: list[str] = [f"shapes train {X_tr.shape} val {X_va.shape}"]

    def _maybe_pca_steps():
        s = [("scaler", StandardScaler())]
        if args.pca and args.pca > 0:
            s.append(("pca", PCA(n_components=min(args.pca, X_tr.shape[1]), random_state=42)))
        return s

    pipe_rbf = Pipeline(_maybe_pca_steps() + [("clf", SVC(kernel="rbf", class_weight="balanced"))])
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", "auto", 1e-3, 1e-2],
    }
    inner_cv = max(2, min(args.rbf_cv, max(len(y_tr) // 3, 2)))
    gs = GridSearchCV(
        pipe_rbf,
        param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    gs.fit(X_tr, y_tr)
    out_lines.append(f"RBF-SVM GridSearchCV cv={inner_cv} best_params={gs.best_params_}")
    out_lines.append(f"best_cv_f1_macro={gs.best_score_:.6f}")
    out_lines.append(_val_report("RBF-SVM (best_estimator_)", y_va, gs.predict(X_va)))

    if args.save_sklearn_bundle:
        joblib.dump(gs.best_estimator_, args.save_sklearn_bundle)
        out_lines.append(f"saved joblib → {args.save_sklearn_bundle}")

    pipe_lin = Pipeline(
        _maybe_pca_steps()
        + [
            (
                "clf",
                LinearSVC(
                    max_iter=8000,
                    dual="auto",
                    class_weight="balanced",
                    random_state=42,
                ),
            )
        ]
    )
    out_lines.append(_fit_report("LinearSVC", pipe_lin, X_tr, y_tr, X_va, y_va))

    pipe_rf = Pipeline(
        _maybe_pca_steps()
        + [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )
    out_lines.append(_fit_report("RandomForest", pipe_rf, X_tr, y_tr, X_va, y_va))

    try:
        from xgboost import XGBClassifier

        n_cls = int(np.max(y_tr)) + 1
        pipe_xgb = Pipeline(
            _maybe_pca_steps()
            + [
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        objective="multi:softprob",
                        num_class=n_cls,
                        eval_metric="mlogloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        )
        out_lines.append(_fit_report("XGBoost", pipe_xgb, X_tr, y_tr, X_va, y_va))
    except ImportError:
        msg = "[train_svm] XGBoost 未安装，已跳过（可选: pip install xgboost）"
        out_lines.append(msg)
        print(msg, file=sys.stderr, flush=True)

    body = "\n".join(out_lines) + "\n"
    _append_log(args.log_file, body)
    print(body, flush=True)


if __name__ == "__main__":
    main()

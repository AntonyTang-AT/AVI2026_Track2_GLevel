"""
使用 Nanbeige4-3B（或其它 HuggingFace CausalLM）对转写文本做离线编码。

支持（与推荐一致）：
  - 池化：mean / last（末 token）/ attn（无参数内容注意力池化）
  - 多层：仅用最后一层，或对最后 K 层做 mean / concat（concat 时输出维 = hidden_size * K）
  - 可选前缀 prompt（适合 Instruct / Thinking 权重）
  - 轻量预处理：空白规范化（非 LLM 纠错）

训练时请根据日志中的「输出维度」设置 --text_dim。

示例：
  python -m features.extract_text \\
    --text_dir /path/to/transcripts \\
    --out_dir /path/to/train_feature/text \\
    --model_id Nanbeige/Nanbeige4-3B-Base \\
    --pooling attn --layer_fuse mean_k --num_last_layers 4

  # Instruct/Thinking 可加引导语：
  # --text_prefix "下面是一段访谈转写，关注情绪与压力相关表述："

依赖：transformers, torch, accelerate（可选）
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


DEFAULT_MODEL = "Nanbeige/Nanbeige4-3B-Base"


@dataclass
class EncodeConfig:
    max_length: int = 2048
    pooling: str = "attn"  # mean | last | attn
    layer_fuse: str = "mean_k"  # last | mean_k | concat_k
    num_last_layers: int = 4
    text_prefix: str = ""
    preprocess: str = "normalize"  # none | normalize


def _preprocess_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "normalize":
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()
    raise ValueError(f"未知 preprocess: {mode}")


def _apply_prefix(text: str, prefix: str) -> str:
    if not (prefix and prefix.strip()):
        return text
    p = prefix.strip()
    if not text.strip():
        return p
    return f"{p}\n\n{text.strip()}"


def _load_causal_lm(
    model_id: str,
    device: torch.device,
    pooling: str,
) -> Tuple[object, torch.nn.Module]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # last token 池化时须左填充，否则 pad 在右侧会污染「最后」位置
    tokenizer.padding_side = "left" if pooling == "last" else "right"

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    try:
        import accelerate  # noqa: F401

        _has_accelerate = True
    except ImportError:
        _has_accelerate = False
    if device.type == "cuda" and _has_accelerate:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device.type == "cuda" and not _has_accelerate:
        model.to(device)
    elif device.type != "cuda":
        model.to(device)

    model.eval()
    return tokenizer, model


def _base_transformer(model: torch.nn.Module):
    if hasattr(model, "model") and model.model is not None:
        return model.model
    raise RuntimeError("当前模型无 .model 子模块，请检查是否为 CausalLM 类架构。")


def _fuse_hidden_layers(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_fuse: str,
    num_last_layers: int,
) -> torch.Tensor:
    """返回 (B, L, H) 或 concat 时 (B, L, K*H)。"""
    if layer_fuse == "last":
        return hidden_states[-1]

    n = len(hidden_states)
    k = max(1, min(num_last_layers, n))
    layers = list(hidden_states[-k:])
    dev = layers[0].device
    layers = [t.to(dev) for t in layers]

    if layer_fuse == "mean_k":
        return torch.stack(layers, dim=0).mean(dim=0)
    if layer_fuse == "concat_k":
        return torch.cat(layers, dim=-1)
    raise ValueError(f"未知 layer_fuse: {layer_fuse}")


def _ensure_batch_seq_hidden(h: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
    """将 hidden 统一为 (B, L, H)。部分模型输出 (L,B,H)、(B,H,L)、(L,H,B) 等。"""
    if h.dim() != 3:
        return h
    sh = tuple(h.shape)
    if batch in sh and seq_len in sh:
        bi = sh.index(batch)
        li = sh.index(seq_len)
        if bi != li:
            hi = ({0, 1, 2} - {bi, li}).pop()
            return h.permute(bi, li, hi).contiguous()
    if h.size(0) == batch and h.size(1) == seq_len:
        return h
    if h.size(0) == seq_len and h.size(1) == batch:
        return h.transpose(0, 1).contiguous()
    if h.size(0) == batch and h.size(2) == seq_len:
        return h.transpose(1, 2).contiguous()
    return h


def _pool_sequence(
    h: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """h: (B, L, D)，返回 (B, D)。"""
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(dtype=h.dtype)
        denom = mask.sum(dim=1).clamp(min=1)
        return (h * mask).sum(dim=1) / denom

    if pooling == "last":
        # 左填充下，最后一个非 pad 位置在每行末尾
        seq_lens = attention_mask.sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0).long()
        b = h.size(0)
        idx = seq_lens.view(b, 1, 1).expand(b, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)

    if pooling == "attn":
        # 以序列 mean 为 query，对位置做 softmax 加权（无训练参数）；用 bmm 避免错误广播
        mask = attention_mask.unsqueeze(-1).to(dtype=h.dtype)
        denom = mask.sum(dim=1).clamp(min=1)
        q_vec = (h * mask).sum(dim=1) / denom  # (B, D)
        scale = h.size(-1) ** 0.5
        scores = torch.bmm(h, q_vec.unsqueeze(-1)).squeeze(-1) / scale  # (B, L)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (h * w).sum(dim=1)

    raise ValueError(f"未知 pooling: {pooling}")


def output_feature_dim(model: torch.nn.Module, cfg: EncodeConfig) -> int:
    h = int(model.config.hidden_size)
    if cfg.layer_fuse == "concat_k":
        return h * max(1, cfg.num_last_layers)
    return h


@torch.inference_mode()
def encode_text_batch(
    texts: List[str],
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    cfg: EncodeConfig,
) -> np.ndarray:
    """
    返回 float32，形状 (B, D)，D 由 hidden_size 与 layer_fuse 决定。
    """
    proc = [_preprocess_text(t, cfg.preprocess) for t in texts]
    proc = [_apply_prefix(t, cfg.text_prefix) for t in proc]

    enc = tokenizer(
        proc,
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    base = _base_transformer(model)
    out = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    h = _fuse_hidden_layers(hs, cfg.layer_fuse, cfg.num_last_layers)
    b_sz, seq_len = input_ids.shape
    h = _ensure_batch_seq_hidden(h, b_sz, seq_len)
    h = h.to(device)
    pooled = _pool_sequence(h, attention_mask, cfg.pooling)
    return pooled.float().cpu().numpy()


def extract_text_cached(
    text_path: str,
    out_npy_path: str,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    cfg: EncodeConfig,
    overwrite: bool = False,
) -> np.ndarray:
    if os.path.isfile(out_npy_path) and not overwrite:
        return np.load(out_npy_path)
    text = _preprocess_text(_read_txt(text_path), cfg.preprocess)
    if not text.strip():
        text = " "
    text = _apply_prefix(text, cfg.text_prefix)
    vec = encode_text_batch([text], tokenizer, model, device, cfg)[0]
    os.makedirs(os.path.dirname(out_npy_path) or ".", exist_ok=True)
    np.save(out_npy_path, vec.astype(np.float32))
    return vec


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def main():
    p = argparse.ArgumentParser(
        description="CausalLM 文本向量提取（池化/多层/前缀/预处理可配）→ .npy"
    )
    p.add_argument("--text_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pattern", type=str, default="*.txt")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--pooling",
        type=str,
        default="attn",
        choices=("mean", "last", "attn"),
        help="序列池化：mean=掩码均值；last=末 token（左填充）；attn=内容注意力加权（推荐默认）",
    )
    p.add_argument(
        "--layer_fuse",
        type=str,
        default="mean_k",
        choices=("last", "mean_k", "concat_k"),
        help="last=仅最后层；mean_k=最后 K 层逐位置均值；concat_k=最后 K 层通道拼接（维数×K）",
    )
    p.add_argument(
        "--num_last_layers",
        type=int,
        default=4,
        help="mean_k/concat_k 时参与融合的后几层，须 <= 模型总层数",
    )
    p.add_argument(
        "--text_prefix",
        type=str,
        default="",
        help=(
            "在转写前拼接的固定引导语；Base 模型可留空。"
            "Instruct/Thinking 可设例如：下面是一段访谈转写，关注情绪与压力："
        ),
    )
    p.add_argument(
        "--preprocess",
        type=str,
        default="normalize",
        choices=("none", "normalize"),
        help="normalize：合并多余空白与换行（非 LLM 纠错）",
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="仅处理前 N 个 txt（排序后，0=不限制）。用于冒烟或排队试跑，避免一次跑全量。",
    )
    args = p.parse_args()

    cfg = EncodeConfig(
        max_length=args.max_length,
        pooling=args.pooling,
        layer_fuse=args.layer_fuse,
        num_last_layers=args.num_last_layers,
        text_prefix=args.text_prefix,
        preprocess=args.preprocess,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(os.path.join(args.text_dir, "**", args.pattern), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.text_dir, args.pattern)))
    if not paths:
        print("No text files found.", file=sys.stderr)
        sys.exit(1)

    if args.max_files and args.max_files > 0:
        paths = paths[: args.max_files]
        print(
            f"[extract_text] --max_files={args.max_files}，本次仅处理 {len(paths)} 个文件",
            file=sys.stderr,
        )

    tokenizer, model = _load_causal_lm(args.model_id, device, pooling=cfg.pooling)
    out_dim = output_feature_dim(model, cfg)
    print(
        f"[extract_text] model={args.model_id} pooling={cfg.pooling} "
        f"layer_fuse={cfg.layer_fuse} K={cfg.num_last_layers} → 向量维数={out_dim} ，"
        f"训练请加: --text_dim {out_dim}",
        file=sys.stderr,
    )

    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        to_compute: List[str] = []
        compute_idx: List[int] = []

        for i, tp in enumerate(batch_paths):
            rel = os.path.relpath(tp, args.text_dir)
            out_path = os.path.join(args.out_dir, os.path.splitext(rel)[0] + ".npy")
            if os.path.isfile(out_path) and not args.overwrite:
                continue
            raw = _read_txt(tp) or " "
            to_compute.append(raw)
            compute_idx.append(i)

        if to_compute:
            vecs = encode_text_batch(to_compute, tokenizer, model, device, cfg)
            for j, idx in enumerate(compute_idx):
                tp = batch_paths[idx]
                rel = os.path.relpath(tp, args.text_dir)
                out_path = os.path.join(args.out_dir, os.path.splitext(rel)[0] + ".npy")
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                np.save(out_path, vecs[j].astype(np.float32))
                print(out_path)


if __name__ == "__main__":
    main()

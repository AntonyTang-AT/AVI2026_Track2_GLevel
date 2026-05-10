"""
VideoMAE 视频特征离线提取，输出 768 维向量（videomae-base hidden）。
依赖：transformers, torch, opencv-python, numpy

示例：
  python -m features.extract_video --video_dir /path/to/videos --out_dir /path/to/npy --pattern "*.mp4"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"


def _load_videomae_encoder(model_id: str, device: torch.device) -> Tuple[object, torch.nn.Module]:
    from transformers import AutoImageProcessor, VideoMAEForVideoClassification

    processor = AutoImageProcessor.from_pretrained(model_id)
    clf = VideoMAEForVideoClassification.from_pretrained(model_id)
    clf.eval()
    encoder = clf.videomae
    encoder.eval()
    encoder.to(device)
    return processor, encoder


def sample_frames_uniform(video_path: str, num_frames: int = 16):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_rgb = []
    if total <= 0:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames_rgb:
            raise RuntimeError(f"No frames in video: {video_path}")
        idx = np.linspace(0, len(frames_rgb) - 1, num_frames).astype(int)
        return [frames_rgb[i] for i in idx]

    idx = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            continue
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(out) < num_frames:
        while len(out) < num_frames and out:
            out.append(out[-1])
    if not out:
        raise RuntimeError(f"No frames sampled: {video_path}")
    return out[:num_frames]


def extract_videomae(
    video_path: str,
    device: Optional[torch.device] = None,
    num_frames: int = 16,
    processor=None,
    encoder=None,
):
    """
    使用预训练 VideoMAE 编码器；对 last_hidden_state 做时间/ patch 维全局平均池化。
    返回 numpy float32，形状 (768,)。可传入已加载的 processor/encoder 以批量加速。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if processor is None or encoder is None:
        processor, encoder = _load_videomae_encoder(MODEL_ID, device)

    frames = sample_frames_uniform(video_path, num_frames=num_frames)
    inputs = processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        out = encoder(pixel_values=pixel_values)
        hidden = out.last_hidden_state
        vec = hidden.mean(dim=1).squeeze(0)

    dim = vec.numel()
    if dim != 768:
        raise ValueError(
            f"VideoMAE hidden size is {dim}, expected 768. "
            "请在训练配置中将 --video_dim 设为实际维度并在模型中保留线性适配层。"
        )

    return vec.detach().float().cpu().numpy()


def extract_videomae_cached(
    video_path: str,
    out_npy_path: str,
    device: Optional[torch.device] = None,
    num_frames: int = 16,
    overwrite: bool = False,
    processor=None,
    encoder=None,
):
    if os.path.isfile(out_npy_path) and not overwrite:
        return np.load(out_npy_path)
    vec = extract_videomae(
        video_path,
        device=device,
        num_frames=num_frames,
        processor=processor,
        encoder=encoder,
    )
    os.makedirs(os.path.dirname(out_npy_path) or ".", exist_ok=True)
    np.save(out_npy_path, vec)
    return vec


def main():
    parser = argparse.ArgumentParser(description="VideoMAE 特征提取并保存为 .npy")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.mp4")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(os.path.join(args.video_dir, "**", args.pattern), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.video_dir, args.pattern)))
    if not paths:
        print("No videos found.", file=sys.stderr)
        sys.exit(1)

    processor, encoder = _load_videomae_encoder(MODEL_ID, device)
    for vp in paths:
        rel = os.path.relpath(vp, args.video_dir)
        out_path = os.path.join(args.out_dir, os.path.splitext(rel)[0] + ".npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            extract_videomae_cached(
                vp,
                out_path,
                device=device,
                num_frames=args.num_frames,
                overwrite=args.overwrite,
                processor=processor,
                encoder=encoder,
            )
            print(out_path)
        except Exception as e:
            print(f"FAIL {vp}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

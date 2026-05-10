#!/usr/bin/env python3
"""从各模态目录任取一个 .npy，打印 shape / dtype / nbytes，用于核对 --audio_dim --video_dim --text_dim。"""
import argparse
import glob
import os

import numpy as np


def one_sample(path_glob: str):
    files = sorted(glob.glob(path_glob))
    if not files:
        return None
    p = files[0]
    a = np.load(p)
    return p, a.shape, a.dtype, a.nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="")
    ap.add_argument("--video_dir", default="")
    ap.add_argument("--text_dir", default="")
    args = ap.parse_args()
    for name, d in ("audio", args.audio_dir), ("video", args.video_dir), ("text", args.text_dir):
        if not d or not os.path.isdir(d):
            print(f"{name}: (跳过，目录未设置或不存在)")
            continue
        pat = os.path.join(d, "*.npy")
        r = one_sample(pat)
        if r is None:
            print(f"{name}: {d} 下无 .npy")
            continue
        p, shape, dtype, nb = r
        flat = int(np.prod(shape)) if shape else 0
        print(f"{name}: file={os.path.basename(p)}")
        print(f"      shape={shape} dtype={dtype} nbytes={nb}  flattened_dim={flat}")


if __name__ == "__main__":
    main()

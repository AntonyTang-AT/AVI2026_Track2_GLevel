#!/usr/bin/env python3
"""打印 CSV 的列名（首行），便于找哪张表含 g_level / 认知标签。"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="+", help="一个或多个 .csv 路径")
    args = ap.parse_args()
    for p in args.csv_path:
        if not os.path.isfile(p):
            print(f"跳过（不存在）: {p}")
            continue
        df = pd.read_csv(p, nrows=0)
        print(f"\n{p}")
        print("  列:", list(df.columns))


if __name__ == "__main__":
    main()

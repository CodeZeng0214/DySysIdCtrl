"""Checkpoint selection: for every training run keep the checkpoint whose
20-episode trailing reward is the highest.

The reward history stored in a checkpoint covers all episodes up to that
checkpoint, so the whole training curve is read from the last readable
checkpoint of the run; the best 20-episode window is located on that curve and
the checkpoint closest to the end of the window is retained.

Output: results/data/selected_checkpoints.csv
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
DATA = os.path.join(OUT, "data")
TABLES = os.path.join(OUT, "tables")
os.makedirs(DATA, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(HERE))

RUNS = {
    "T1":   "T1_20260327中午MLP无延迟初位移条件部分观测",
    "T1-1": "T1-1_20260328中午MLP无延迟初位移条件部分观测",
    "T1-2": "T1-2_20260407下午MLP无延迟初位移条件部分观测",
    "T2":   "T2_20260317下午无延迟初位移条件部分观测，修改平滑奖励函数，不看动作",
    "T2-1": "T2-1_20260926_随机种子实验T2-4",
    "T2-2": "T2-2_20260928_随机种子实验T2-7",
    "T3":   "T3_20260401下午MLP-3-1延迟初位移条件部分观测",
    "T3-1": "T3-1_20260925_随机种子实验T3-1",
    "T3-2": "T3-2_20260928_随机种子实验T3-2",
    "T4":   "T4_20260324下午-4-1延迟初位移条件部分观测平滑奖励看动作",
    "T4-1": "T4-1_20260928_随机种子实验T4-3",
    "T4-2": "T4-2_20260928_随机种子实验T4-4",
}
WINDOW = 20


def episodes(fname: str) -> int:
    m = re.search(r"ep(\d+)ckpt", fname)
    return int(m.group(1)) if m else -1


def main():
    rows = []
    for run, folder in RUNS.items():
        ck_dir = os.path.join(ROOT, "savedata", folder, "checkpoints")
        files = sorted((f for f in os.listdir(ck_dir) if f.endswith(".pth")), key=episodes)

        pay = None
        for f in reversed(files):
            try:
                pay = torch.load(os.path.join(ck_dir, f), map_location="cpu", weights_only=False)
                break
            except Exception:
                continue
        hist = np.array(pay["history"]["metrics"]["simu_reward_history"], dtype=float)
        w = min(WINDOW, len(hist))
        roll = np.convolve(hist, np.ones(w) / w, mode="valid")
        best = float(roll.max())
        ep_best = int(roll.argmax() + w)          # end of the best window

        chosen = min(((abs(episodes(f) - ep_best), f) for f in files))[1]
        for _, f in sorted(((abs(episodes(f) - ep_best), f) for f in files)):
            try:
                torch.load(os.path.join(ck_dir, f), map_location="cpu", weights_only=False)
                chosen = f
                break
            except Exception:
                continue

        rows.append(dict(run=run, ckpt_ep=episodes(chosen),
                         best20=round(best, 2), window_end=ep_best,
                         path=os.path.join("savedata", folder, "checkpoints", chosen)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "selected_checkpoints.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

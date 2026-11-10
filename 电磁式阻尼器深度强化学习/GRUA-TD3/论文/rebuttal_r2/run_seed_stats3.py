"""Condition tables, part B: three training runs x ten paired realisations.

Checkpoints are the ones selected by select_ckpt.py (best 20-episode training
reward of each run).  Protocol of the manuscript: sampling jitter on in every
condition, loop delay -(3,1), control engaged at 0.3 T, every controller and
the no-control baseline sharing the same delay / jitter realisations.

Output: results/data/exp2c_episodes.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("EVAL_DEVICE", "cpu")

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E
import paired

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
DATA = os.path.join(OUT, "data")
TABLES = os.path.join(OUT, "tables")
os.makedirs(DATA, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)
K = 10
DELAY = 3

GROUPS = {
    "MLP-TD3 (ideal)":  ["T1", "T1-1", "T1-2"],
    "GRUA-TD3 (ideal)": ["T2", "T2-1", "T2-2"],
    "MLP-TD3 (delay)":  ["T3", "T3-1", "T3-2"],
    "GRUA-TD3 (delay)": ["T4", "T4-1", "T4-2"],
}
RUN_OF = {r: g for g, rs in GROUPS.items() for r in rs}
ENV_OF = {r: ("ideal" if "ideal" in g else "delay") for r, g in RUN_OF.items()}


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _run(task):
    name, path, score, cond, delay, seeds, nc = task
    ctrl = E.load_td3(os.path.join(E.ROOT, path))
    dim = ctrl.actor.net[0].in_features if ctrl.arch == "mlp" else ctrl.actor.gru_predictor.gru.input_size
    rows = []
    for s in seeds:
        env, _ = E.env_setting(cond, delay, dim, dt_noise=True, delay_mean=DELAY)
        ctrl.reset()
        peak = E.mean_peak_amplitude(paired.rollout_paired(env, ctrl, s, delay_mean=DELAY)[:, 3])
        a_nc = nc[(cond, delay, s)]
        rows.append(dict(run=name, group=RUN_OF[name], condition=cond,
                         environment="delay" if delay else "ideal", episode=s,
                         nc=a_nc, peak=peak, att=100 * (1 - peak / a_nc), train_score=score))
    return rows


def build_nc(seeds):
    nc = {}
    for cond in E.CONDITIONS:
        for delay in (False, True):
            for s in seeds:
                env, _ = E.env_setting(cond, delay, 4, dt_noise=True, delay_mean=DELAY)
                nc[(cond, delay, s)] = E.mean_peak_amplitude(
                    paired.rollout_paired(env, None, s, delay_mean=DELAY)[:, 3])
    return nc


def main(workers=5):
    sel = pd.read_csv(os.path.join(DATA, "selected_checkpoints.csv"))
    seeds = list(range(K))
    ncs = build_nc(seeds)
    tasks = [(r.run, r.path, r.best20, c, ENV_OF[r.run] == "delay", seeds, ncs)
             for r in sel.itertuples() for c in E.CONDITIONS]

    rows = []
    with Pool(workers, initializer=_init_worker) as pool:
        for part in tqdm(pool.imap_unordered(_run, tasks), total=len(tasks),
                         desc="conditions", unit="blk", ncols=88):
            rows += part
    eps = pd.DataFrame(rows)
    eps.to_csv(os.path.join(DATA, "exp2c_episodes.csv"), index=False)

    sm = (eps.groupby(["group", "run", "condition", "environment"], sort=False)
             .agg(nc=("nc", "mean"), peak=("peak", "mean"), peak_sd=("peak", "std"),
                  att=("att", "mean"), att_sd=("att", "std"),
                  train_score=("train_score", "first")).reset_index())
    sm.to_csv(os.path.join(DATA, "exp2c_summary.csv"), index=False)
    print(sm.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

"""Robustness with the parameter definition of the manuscript (M and k_M only)
and paired realisations - verification against 论文/other-data.

Reference values (other-data/robustness_summary.csv):
    initial displacement: 92.24 / 96.54 / 76.18 / 80.02 / 94.50 %
    30 Hz              : 88.47 / 79.15 / 78.69 / 72.77 / 83.74 %
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
os.makedirs(OUT, exist_ok=True)

K = 10
DELAY = 4
NOMINAL = {'m': 1.0, 'M': 15.0, 'k_m': 30_000.0, 'k_M': 300_000.0,
           'k_f': 100.0, 'c_m': 0.0001, 'c_M': 0.001}
CONFIGS = [("Nominal", None),
           ("0.9M", {"M": 13.5}),
           ("1.1M", {"M": 16.5}),
           ("0.9K", {"k_M": 270_000.0}),
           ("1.1K", {"k_M": 330_000.0})]
CONDS = [("free", 1.2, 0.006), ("fixed", 2.5, 0.0)]
CKPT = os.environ.get("ROB_CKPT", "savedata/A消融实验/T4-2-GRUA基准-0928_204140_ep200ckpt.pth")


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _params(over):
    p = dict(NOMINAL)
    if over:
        p.update(over)
    return p


def _run(task):
    cond, name, over, seeds = task
    sp = _params(over)
    ctrl = E.load_td3(CKPT)
    rows = []
    for s in seeds:
        env_nc, _ = E.env_setting(cond, True, 5, sys_params=sp, dt_noise=True, delay_mean=DELAY)
        a_nc = E.mean_peak_amplitude(paired.rollout_paired(env_nc, None, s, delay_mean=DELAY)[:, 3])
        env_c, _ = E.env_setting(cond, True, 5, sys_params=sp, dt_noise=True, delay_mean=DELAY)
        ctrl.reset()
        a_c = E.mean_peak_amplitude(paired.rollout_paired(env_c, ctrl, s, delay_mean=DELAY)[:, 3])
        rows.append(dict(condition=cond, config=name, episode=s, nc=a_nc, ctrl=a_c,
                         att=100 * (1 - a_c / a_nc)))
    return rows


def main(workers=5, tag=""):
    global K
    K = int(os.environ.get("ROB_K", K))
    seeds = list(range(K))
    tasks = [(c, n, o, seeds) for c in ("free", "fixed") for n, o in CONFIGS]
    rows = []
    with Pool(workers, initializer=_init_worker) as pool:
        for part in tqdm(pool.imap_unordered(_run, tasks), total=len(tasks),
                         desc="robustness", unit="blk", ncols=88):
            rows += part

    eps = pd.DataFrame(rows)
    eps.to_csv(os.path.join(DATA, f"exp3b_episodes{tag}.csv"), index=False)
    sm = (eps.groupby(["condition", "config"], sort=False)
             .agg(nc=("nc", "mean"), nc_sd=("nc", "std"),
                  ctrl=("ctrl", "mean"), ctrl_sd=("ctrl", "std"),
                  att=("att", "mean"), att_sd=("att", "std")).reset_index())
    sm["att_ratio"] = 100 * (1 - sm.ctrl / sm.nc)
    sm.to_csv(os.path.join(DATA, f"exp3b_summary{tag}.csv"), index=False)

    ref = {("free", "Nominal"): 92.24, ("free", "0.9M"): 96.54, ("free", "1.1M"): 76.18,
           ("free", "0.9K"): 80.02, ("free", "1.1K"): 94.50,
           ("fixed", "Nominal"): 88.47, ("fixed", "0.9M"): 79.15, ("fixed", "1.1M"): 78.69,
           ("fixed", "0.9K"): 72.77, ("fixed", "1.1K"): 83.74}
    sm["other"] = [ref[(r.condition, r.config)] for r in sm.itertuples()]
    sm["diff"] = sm.att - sm.other
    print(sm.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

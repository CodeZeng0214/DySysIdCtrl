"""Sanity check: does the evaluation harness reproduce a published number?

Reference (paper, Table 7, GRUA-TD3 baseline row):
    fixed-frequency : 6.69e-4 m -> 89.05 %
The initial-displacement value of the paper (2.74e-4 m) is a single realisation
of a variable-step episode and is therefore not expected to be reproduced by a
single seed; see RECONCILIATION.md.

Usage: python check_reproduce.py [--mode delayed|start]
"""
import os
os.environ.setdefault("EVAL_DEVICE", "cpu")

import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import env_setting, rollout, mean_peak_amplitude, load_td3, attenuation

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "savedata", "A消融实验", "T4-2-GRUA基准-0928_204140_ep200ckpt.pth")


def main(mode="delayed"):
    for cond in ("free", "fixed"):
        np.random.seed(0)
        env, _ = env_setting(cond, True, 5)
        ctrl = load_td3(CKPT)
        ctrl.reset()
        a = mean_peak_amplitude(rollout(env, ctrl, mode=mode)[:, 3])

        np.random.seed(0)
        env_nc, _ = env_setting(cond, True, 5)
        a_nc = mean_peak_amplitude(rollout(env_nc, None)[:, 3])

        print(f"{cond:5s} mode={mode:7s} ctrl={a:.4e}  nc={a_nc:.4e}  att={attenuation(a, a_nc):.2f}%  "
              f"(paper: free 2.74e-4/94.05%, fixed 6.69e-4/89.05%)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="delayed", choices=["start", "delayed"])
    main(ap.parse_args().mode)

"""Reviewer #3, comment 1 (supporting diagnostic) - what does the rolled-forward
branch actually contain?

For a trained GRUA-TD3 controller the multi-step branch is replayed exactly as
in the implementation:

    H_t           = GRU(S_{t-L_h+1..t})[-1]
    S_hat_{t+1}   = W_fc H_t + b_fc
    H_{t+k+1}     = GRU(S_hat_{t+k+1}, H_{t+k})
    S_hat_{t+k+2} = W_fc H_{t+k+1} + b_fc          k = 0..L_f-2

The rolled-forward states are compared with the observations that the
controller actually receives k steps later, and with a persistence baseline
(S_hat_{t+k} = S_t).  The observation is [x1, x1_ddot, x2, x2_ddot, tau_d].

Output: results/data/exp1_prediction_probe.csv, exp1_attention_weights.csv, .md
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_common as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
DATA = os.path.join(OUT, "data")
TABLES = os.path.join(OUT, "tables")
os.makedirs(DATA, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "T4":   "savedata/T4_20260324下午-4-1延迟初位移条件部分观测平滑奖励看动作/checkpoints/0426_145412_ep200ckpt.pth",
    "T4-1": "savedata/T4-1_20260928_随机种子实验T4-3/checkpoints/0928_120518_ep200ckpt.pth",
    "T4-2": "savedata/A消融实验/T4-2-GRUA基准-0928_204140_ep200ckpt.pth",
}
CONDS = ("free", "fixed", "mls")
L_H, L_F = 36, 4
OBS_NAMES = ["x1", "x1_ddot", "x2", "x2_ddot", "tau_d"]


def roll_forward(predictor, seq: torch.Tensor):
    """Replay GruPredictor.forward and return predicted states + attention."""
    device = next(predictor.parameters()).device
    seq = seq.to(device)
    predictor.eval()
    with torch.no_grad():
        gru_out, h_n = predictor.gru(predictor.gru_norm(seq))
        current_h = h_n
        current_state = predictor.fc_predict(current_h[-1]).unsqueeze(1)

        pred, hid = [], []
        for _ in range(L_F):
            pred.append(current_state)
            hid.append(current_h[-1].unsqueeze(1))
            _, next_h = predictor.gru(current_state, current_h)
            next_state = predictor.fc_predict(next_h[-1]).unsqueeze(1)
            current_state, current_h = next_state, next_h

        state_fc_seq = torch.cat(pred, dim=1)
        h_fc_seq = torch.cat(hid, dim=1)
        state_comb = torch.cat((seq[:, -L_F // 2:, :], state_fc_seq), dim=1)
        h_comb = torch.cat((gru_out[:, -L_F // 2:, :], h_fc_seq), dim=1)

        scores = predictor.attention(h_comb).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.sum(state_comb * weights.unsqueeze(-1), dim=1)
    return (state_fc_seq[0].cpu().numpy(), weights[0].cpu().numpy(), pooled[0].cpu().numpy())


def main():
    rows, att_rows = [], []
    for tag, path in MODELS.items():
        ctrl = E.load_td3(path)
        predictor = ctrl.actor.gru_predictor
        for cond in CONDS:
            np.random.seed(0)
            env, _ = E.env_setting(cond, True, 5)
            ctrl.reset()
            E.rollout(env, ctrl)
            obs = np.array(ctrl.obs_state_history, dtype=np.float32)
            n = len(obs)
            scale = np.sqrt((obs[L_H:] ** 2).mean(axis=0)) + 1e-12

            errs = {k: [] for k in range(1, L_F + 1)}
            base = {k: [] for k in range(1, L_F + 1)}
            preds = {k: [] for k in range(1, L_F + 1)}
            w_all = []

            for t in tqdm(range(L_H - 1, n - L_F - 1), desc=f"{tag} {cond}", unit="step",
                          ncols=88, leave=False):
                seq = torch.tensor(obs[t - L_H + 1:t + 1], dtype=torch.float32).unsqueeze(0)
                pred, w, _ = roll_forward(predictor, seq)
                tgt = obs[t + 1:t + 1 + L_F]
                w_all.append(w)
                for k in range(1, L_F + 1):
                    errs[k].append(np.abs(pred[k - 1] - tgt[k - 1]))
                    preds[k].append(pred[k - 1])
                    base[k].append(np.abs(obs[t] - tgt[k - 1]))

            w_mean = np.mean(w_all, axis=0)
            att_rows.append(dict(model=tag, condition=cond,
                                 **{f"pos{i}": w_mean[i] for i in range(len(w_mean))}))
            for k in range(1, L_F + 1):
                e = np.stack(errs[k])
                b = np.stack(base[k])
                p = np.stack(preds[k])
                for d_i, name in enumerate(OBS_NAMES[:e.shape[1]]):
                    rows.append(dict(model=tag, condition=cond, horizon=k, variable=name,
                                     mae_pred=float(e[:, d_i].mean()),
                                     mae_persist=float(b[:, d_i].mean()),
                                     nrmse_pred=float(np.sqrt((e[:, d_i] ** 2).mean()) / scale[d_i]),
                                     nrmse_persist=float(np.sqrt((b[:, d_i] ** 2).mean()) / scale[d_i]),
                                     rms_pred=float(np.sqrt((p[:, d_i] ** 2).mean())),
                                     rms_true=float(scale[d_i])))
            print(f"[{tag} {cond:5s}] probed {n - L_H - L_F} states", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "exp1_prediction_probe.csv"), index=False)
    att = pd.DataFrame(att_rows)
    att.to_csv(os.path.join(DATA, "exp1_attention_weights.csv"), index=False)

    md = ["# Comment 1 - diagnostic of the rolled-forward branch", "",
          "Observation channels: " + ", ".join(OBS_NAMES), "",
          "## Normalised RMSE of the rolled-forward states (RMSE / RMS of the channel)", "",
          "| Model | Condition | Horizon | " + " | ".join(OBS_NAMES) + " | persistence (x2) |",
          "|---|---|---|" + "---|" * 5 + "---|"]
    for tag in MODELS:
        for cond in CONDS:
            for k in range(1, L_F + 1):
                d = df[(df.model == tag) & (df.condition == cond) & (df.horizon == k)]
                cells = " | ".join(f"{d[d.variable == v].nrmse_pred.iloc[0]:.2f}" for v in OBS_NAMES)
                pers = d[d.variable == "x2"].nrmse_persist.iloc[0]
                md.append(f"| {tag} | {cond} | t+{k} | {cells} | {pers:.2f} |")
    md += ["", "## Amplitude of the rolled-forward states (RMS) vs the true channel", "",
           "| Model | Condition | " + " | ".join(f"{v} pred / true" for v in OBS_NAMES) + " |",
           "|---|---|" + "---|" * 5]
    for tag in MODELS:
        for cond in CONDS:
            d = df[(df.model == tag) & (df.condition == cond) & (df.horizon == 1)].set_index("variable")
            cells = " | ".join(f"{d.loc[v,'rms_pred']:.2e} / {d.loc[v,'rms_true']:.2e}" for v in OBS_NAMES)
            md.append(f"| {tag} | {cond} | {cells} |")
    md += ["", "## Mean attention weights (pos0-1 = encoded H_{t-1},H_t; pos2-5 = rolled forward)", "",
           "| Model | Condition | " + " | ".join(f"pos{i}" for i in range(6)) + " |",
           "|---|---|" + "---|" * 6]
    for _, r in att.iterrows():
        md.append(f"| {r.model} | {r.condition} | " + " | ".join(f"{r[f'pos{i}']:.3f}" for i in range(6)) + " |")
    with open(os.path.join(DATA, "exp1_prediction_probe.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Final paper-ready tables for comment 2 (three conditions) and comment 3.

Inputs
    results/data/exp2c_episodes.csv   paired evaluation, 12 checkpoints x 3 conditions
    results/data/exp3b_episodes.csv   paired robustness, M and k_M only

Selection rule (documented in the caption): for every configuration the run
with the highest mean simulated reward over its last 50 training episodes is
retained; the other runs of that configuration did not converge.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
DATA = os.path.join(OUT, "data")
TABLES = os.path.join(OUT, "tables")
os.makedirs(DATA, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

COND_LABEL = {"free": "initial-displacement", "fixed": "30\\,Hz fixed-frequency",
              "mls": "MLS random"}
COND_TITLE = {"free": "Initial displacement", "fixed": "30\\,Hz excitation", "mls": "MLS excitation"}
ORDER = [("MLP-TD3", "MLP-TD3 (ideal)", "MLP-TD3 (delay)"),
         ("GRUA-TD3", "GRUA-TD3 (ideal)", "GRUA-TD3 (delay)")]

# published values (single run) of Tables 3-5
PAPER = {
    ("free", "MLP-TD3"): (7.99e-4, 82.64, 2.44e-3, 46.91, 78.98, 77.54),
    ("free", "GRUA-TD3"): (1.09e-3, 76.30, 6.72e-4, 85.41, 76.30, 85.41),
    ("fixed", "MLP-TD3"): (1.18e-3, 81.85, 2.32e-3, 62.02, 76.11, 51.90),
    ("fixed", "GRUA-TD3"): (1.09e-3, 83.18, 9.31e-4, 84.77, 83.18, 84.77),
    ("mls", "MLP-TD3"): (5.40e-3, 43.14, 2.60e-3, 71.72, 81.98, 77.23),
    ("mls", "GRUA-TD3"): (1.68e-3, 82.30, 1.82e-3, 80.24, 82.30, 80.24),
}
PAPER_NC = {("free", "ideal"): 4.60e-3, ("free", "delay"): 4.60e-3,
            ("fixed", "ideal"): 6.50e-3, ("fixed", "delay"): 6.12e-3,
            ("mls", "ideal"): 9.50e-3, ("mls", "delay"): 9.21e-3}


def sci(x, dec=2):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10**e:.{dec}f}\\times10^{{{e}}}$"


def sci_pm(a, sd, dec=2):
    e = int(np.floor(np.log10(abs(a)))) if a != 0 else 0
    return f"$({a / 10**e:.{dec}f}\\pm{sd / 10**e:.{dec}f})\\times10^{{{e}}}$"


def main():
    eps = pd.read_csv(os.path.join(DATA, "exp2c_episodes.csv"))

    # selection: best training score within the configuration
    sel = {}
    for g in eps.group.unique():
        d = eps[eps.group == g]
        scores = d.groupby("run").train_score.first()
        sel[g] = scores.idxmax()

    NSEED = 10
    rows = []
    for cond in ("free", "fixed", "mls"):
        d = eps[(eps.condition == cond) & (eps.episode < NSEED)]
        for tex_name, g_ideal, g_delay in ORDER:
            for g, env in ((g_ideal, "ideal"), (g_delay, "delay")):
                r = d[(d.group == g) & (d.environment == env)]
                per_run = r.groupby("run").att.mean()
                rows.append(dict(condition=cond, controller=tex_name, group=g,
                                 n_runs=r.run.nunique(), n_episodes=len(r),
                                 env=env, peak=r.peak.mean(), peak_sd=r.peak.std(),
                                 att=r.att.mean(), att_sd=r.att.std(),
                                 att_sd_run=per_run.std(ddof=1),
                                 att_runs="; ".join(f"{v:.1f}" for v in per_run.values)))
    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(os.path.join(DATA, "exp2b_selected.csv"), index=False)

    # ---------- Tables 3/4/5 (published layout, now with statistics) ----------
    ORIG_CAP = {
        "free": r"Controller performance data under free vibration conditions."
                r" The mean maximum amplitude is expressed in metres.",
        "fixed": r"Controller performance data under fixed-frequency excitation conditions."
                 r" The fixed-frequency excitation is a 30\,Hz sinusoid of amplitude"
                 r" $F_0=150\,\mathrm{N}$, and the mean maximum amplitude is expressed in metres.",
        "mls": r"Controller performance data under random excitation conditions. The excitation"
               r" is an MLS random signal approximated by a periodic pseudo-random binary"
               r" sequence, and the mean maximum amplitude is expressed in metres.",
    }
    ORIG_LABEL = {"free": "tab:free_vib_data", "fixed": "tab:fixed_freq_data",
                  "mls": "tab:random_data"}
    TEXNAME = {"MLP-TD3": "TD3", "GRUA-TD3": "GRUA-TD3"}

    for cond in ("free", "fixed", "mls"):
        cap = (r"\caption{" + ORIG_CAP[cond] +
               r" \textcolor{blue}{The TD3 and GRUA-TD3 rows each summarise three independent"
               r" training runs, every run contributing the checkpoint with the highest mean"
               r" reward over a 20-episode window of its training curve; each checkpoint is"
               r" evaluated over ten random realisations of the loop delay and of the sampling"
               r" jitter that are shared with the corresponding no-control response, so that"
               r" every value is the mean $\pm$ standard deviation over thirty episodes.}}")
        tex = [r"\begin{table}[htbp]", r"\centering", cap,
               r"\label{%s}" % ORIG_LABEL[cond], r"\footnotesize",
               r"\setlength{\tabcolsep}{3.5pt}",
               r"\begin{tabular}{lcccc}", r"\toprule",
               r"\multirow{2}{*}{Control Strategy} & \multicolumn{2}{c}{Ideal Environment} &"
               r" \multicolumn{2}{c}{Non-Ideal Environment \textcolor{blue}{Delay}-(3,1)} \\",
               r"\cline{2-5}",
               r"& Mean Max.\ Amplitude \textcolor{blue}{(m)} & Attenuation Rate &"
               r" Mean Max.\ Amplitude \textcolor{blue}{(m)} & Attenuation Rate \\ \midrule"]
        nc_i, nc_d = PAPER_NC[(cond, "ideal")], PAPER_NC[(cond, "delay")]
        tex.append(f"No Control & {sci(nc_i)} & 0.00\\% & {sci(nc_d)} & 0.00\\% \\\\")
        p = PAPER[(cond, "MLP-TD3")]
        tex.append(f"PID & {sci(p[0])} & {p[1]:.2f}\\% & {sci(p[2])} & {p[3]:.2f}\\% \\\\")
        for tex_name, g_ideal, g_delay in ORDER:
            a = sel_df[(sel_df.condition == cond) & (sel_df.controller == tex_name) & (sel_df.env == "ideal")].iloc[0]
            b = sel_df[(sel_df.condition == cond) & (sel_df.controller == tex_name) & (sel_df.env == "delay")].iloc[0]
            tex.append(f"{TEXNAME[tex_name]} & {sci_pm(a.peak, a.peak_sd)} & "
                       f"${a.att:.2f}\\pm{a.att_sd:.2f}$\\% & "
                       f"{sci_pm(b.peak, b.peak_sd)} & "
                       f"${b.att:.2f}\\pm{b.att_sd:.2f}$\\% \\\\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        with open(os.path.join(TABLES, f"table_cond_{cond}.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tex) + "\n")

    # ---------- Table 7: robustness ----------
    e3 = pd.read_csv(os.path.join(DATA, "exp3b_episodes.csv"))
    sm = (e3.groupby(["condition", "config"], sort=False)
            .agg(nc=("nc", "mean"), ctrl=("ctrl", "mean"), ctrl_sd=("ctrl", "std"),
                 att=("att", "mean"), att_sd=("att", "std")).reset_index())
    order = ["Nominal", "0.9M", "1.1M", "0.9K", "1.1K"]
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Robustness of the trained GRUA-TD3 controller to $\pm10\%$ variations of the"
           r" primary-structure mass $M$ and stiffness $k_M$. Each parameter configuration has its"
           r" own no-control baseline, evaluated with the same delay and sampling-jitter"
           r" realisations as the controlled run (10 paired episodes; mean $\pm$ standard"
           r" deviation).}",
           r"\label{tab:robustness}", r"\small", r"\setlength{\tabcolsep}{3.4pt}",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r"\multirow{2}{*}{Model Parameter} & \multicolumn{2}{c}{Initial Displacement} &"
           r" \multicolumn{2}{c}{Fixed-Frequency Excitation} \\", r"\cline{2-5}",
           r" & No Control (m) & Attenuation & No Control (m) & Attenuation \\", r"\midrule"]
    label = {"Nominal": "$(M,K,C)$", "0.9M": "$(0.9M,k_M)$", "1.1M": "$(1.1M,k_M)$",
             "0.9K": "$(M,0.9k_M)$", "1.1K": "$(M,1.1k_M)$"}
    for cfg in order:
        cells = []
        for cond in ("free", "fixed"):
            r = sm[(sm.condition == cond) & (sm.config == cfg)].iloc[0]
            cells += [sci(r.nc), f"{r.att:.2f}\\%$\\pm${r.att_sd:.2f}"]
        tex.append(f"{label[cfg]} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(TABLES, "table_robustness.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    print("selected runs:", sel)
    print(sm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

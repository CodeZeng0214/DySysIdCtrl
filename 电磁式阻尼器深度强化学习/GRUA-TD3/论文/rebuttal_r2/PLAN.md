# 修改版论文施工计划（Round 2 审稿回复）

> 基准文件：`论文/Manuscript-v2/ManuscriptZ.tex`（行号按当前版本，定位以 `\label` 为准）
> **第二版变更**：三张工况表由「新增」改为「替换原表」，取消新增 4.4.6 节；组间 SD 不进表格，改在正文说明。
> 状态：**仅计划，尚未改动 tex**

---

## 一、本轮结果文件说明

### 1.1 脚本（可复现链）

| 文件 | 作用 | 对应意见 |
|---|---|---|
| `eval_common.py` | 环境构造、峰值指标（后 30% 窗口内局部峰均值）、按 `state_dim` 自动匹配的 checkpoint 加载 | 全部 |
| `paired.py` | 配对噪声：预生成 `delay/dt` 序列并注入受控与无控两条轨迹 | ②③ |
| `select_ckpt.py` | 选点：每个训练 run 取「训练奖励最佳 20 回合窗口」的 checkpoint（纯训练期判据） | ② |
| `run_seed_stats3.py` | 12 模型 × 3 工况 × 10 次配对实现 | ② |
| `run_robustness2.py` | 5 组参数 × 2 工况 × (无控制 + 受控) × 10 次配对实现，只扰动 `M`、`k_M` | ③ |
| `make_tables_final.py` | 生成 Table 3/4/5（合并版）与 Table 7 的 LaTeX | ②③ |
| `run_prediction_probe.py` | roll-forward 输出与真实观测的多步误差、注意力权重分布 | ①（仅回复信） |
| `check_reproduce.py` | 复现校验：定频 6.6967e-4 / 89.05%，与论文原值一致 | 全部 |

### 1.2 数据与成品

| 文件 | 内容 |
|---|---|
| `results/data/exp2c_episodes.csv` | 三工况逐回合结果（`run, group, condition, environment, episode, nc, peak, att, train_score`），360 行 |
| `results/data/exp2b_selected.csv` | 表 3/4/5 的中间汇总，含 `att_sd_run`（组间 SD）与 `att_runs`（三个 run 各自均值） |
| `results/data/selected_checkpoints.csv` | 12 个 run 选中的 ckpt 与训练奖励 |
| `results/data/exp3b_episodes.csv` / `exp3b_summary.csv` | 鲁棒性逐回合 / 汇总（含每组独立 NC、`nc_sd`） |
| **`results/tables/table_cond_{free,fixed,mls}.tex`** | **Table 3/4/5 成品**（5 列，与原表同构，ML 行改为 mean±SD） |
| **`results/tables/table_robustness.tex`** | **Table 7 成品**（7 列，含每组独立 NC） |
| `results/data/exp1_*` | 预测诊断数据（NRMSE≈1） |
| `preview/table_cond_preview.pdf` | 三张合并表的实际编译预览（已验证 0 错误、0 Overfull） |
| `RECONCILIATION.md` | 论文原表数值与仓库 CSV 的逐格溯源记录 |

### 1.3 统计口径（正文表述必须与之一致）

- 每个 ML 配置 = **3 个独立训练 run**；每个 run 取训练奖励最佳 20 回合窗口的 ckpt；
- 每个 ckpt 评估 **10 次配对实现**（delay + jitter 序列与对应无控制响应共用）；
- 表格中的 `±` = **30 个回合**的标准差；PID 与 No Control 仍为单值（确定性 / 单一实现）；
- 组间波动（3 个 run 均值之间的 SD）**不进表格**，在 4.4 开头的指标说明里用一句话交代（GRUA/TD3：初位移 4.1/8.6、30 Hz 0.7/1.1、MLS 2.9/2.5；非理想环境 11.0/27.4 与 7.5/29.1）。

---

## 二、表格方案（全部替换，不新增表）

### 2.1 Table 3 —— 初位移

- **源文件**：`results/tables/table_cond_free.tex`
- **位置**：第 744–758 行，`\label{tab:free_vib_data}`，**整块替换**
- **结构**：5 列不变（Control Strategy / 理想幅值 / 理想衰减率 / 延迟幅值 / 延迟衰减率），行仍为 No Control / PID / TD3 / GRUA-TD3
- **数值**：TD3 63.90±10.19（理想）、57.41±28.42（延迟）；GRUA-TD3 83.05±6.34、80.88±10.70
- **caption**：在原句后追加一句（已标蓝），说明 3 次训练 × 10 次实现 = 30 回合的 mean±SD

### 2.2 Table 4 —— 30 Hz 定频

- **源文件**：`results/tables/table_cond_fixed.tex`
- **位置**：第 791–806 行，`\label{tab:fixed_freq_data}`，整块替换
- **注意**：原表数据行（802–804）当前带 `\textcolor{blue}{}`，替换时统一去掉（数值已明显变化，靠 ± 自证），只保留 caption 的蓝色新增句
- **数值**：TD3 72.28±4.83、62.39±5.09；GRUA-TD3 82.20±4.12、86.08±2.48

### 2.3 Table 5 —— MLS 随机激励

- **源文件**：`results/tables/table_cond_mls.tex`
- **位置**：第 833–848 行，`\label{tab:random_data}`，整块替换
- **数值**：TD3 79.99±3.41、46.93±31.01；GRUA-TD3 81.59±7.05、77.77±6.27

### 2.4 Table 7 —— 鲁棒性

- **源文件**：`results/tables/table_robustness.tex`
- **位置**：第 898–918 行，`\label{tab:robustness}`，整块替换
- **结构**：5 列 → **7 列**（每个工况下增加 `No Control (m)`）
- **必修**：行标签 `$(0.9M,k_M)$` 改为 `$(0.9M,K,C)$`，与正文第 896 行命名统一
- **新数值**

| 参数组 | 初位移 NC | 初位移衰减率 | 30 Hz NC | 30 Hz 衰减率 |
|---|---|---|---|---|
| (M,K,C) | 4.60e-3 | 94.72 ± 1.26 | 5.84e-3 | 87.91 ± 1.21 |
| (0.9M,K,C) | 4.52e-3 | 97.46 ± 1.10 | 4.14e-3 | 77.71 ± 0.23 |
| (1.1M,K,C) | 4.90e-3 | 84.37 ± 7.50 | 2.57e-3 | 78.00 ± 0.04 |
| (M,0.9K,C) | 4.90e-3 | 86.60 ± 2.31 | 2.19e-3 | 71.77 ± 0.05 |
| (M,1.1K,C) | 4.51e-3 | 96.04 ± 2.87 | 4.70e-3 | 82.77 ± 0.77 |

---

## 三、正文修改清单

### 3.1 4.4「Results Analysis」开头（第 712 行后）

在三个工况的定义之后、4.4.1 之前，**新增 3 句**指标说明：

1. 表格中 ML 两行现在是 3 次训练 × 10 次配对实现的统计量；
2. `±` 是 30 个回合的标准差，主要反映同一次训练在不同延迟/抖动实现下的波动；
3. **组间波动**（三个训练 run 之间的标准差）在其中占比更小——初位移 4.1/8.6、30 Hz 0.7/1.1、MLS 2.9/2.5（GRUA/TD3）——只有非理想环境下达 11.0 vs 27.4，说明 GRUA 对训练初始条件的依赖更弱。

### 3.2 4.4.1 初位移（第 767 行）

- “stably exceeds 85%” → **“remains close to 81 % (Table 3), with a run-to-run spread of about 12 percentage points”**（新值 80.88 ± 10.70）
- 第 765 行的 “exceed 75%” 指固定 3 步延迟的图（`fig:free_vib_delay_30`），不是表中 -(3,1)，保留

### 3.3 4.4.2 30 Hz（第 774、776 行）

| 行 | 现值 | 新值 |
|---|---|---|
| 774 | PID 81.85 / TD3 76.11 / GRUA 83.18 | PID **81.85**（不变）/ TD3 **72.28 ± 4.83** / GRUA **82.20 ± 4.12** |
| 776 | PID 62.02 / TD3 51.90 / GRUA 84.77 | PID **62.02**（不变）/ TD3 **62.39 ± 5.09** / GRUA **86.08 ± 2.48** |

> TD3 延迟由 51.90 升到 62.39，原文 “loses a considerable part of its effectiveness” 要弱化为 “drops by about ten points”，但结论方向不变。

### 3.4 4.4.3 MLS（第 814 行）

- “reducing the vibration amplitude by over 80%” → **“by approximately 80%”**（GRUA 理想 81.59、延迟 77.77）

### 3.5 4.4.5 鲁棒性（第 920 行，整段重写）

必须替换的旧数字：89.71–95.56 %、SD 1.97 %、86.08–91.01 %、SD 1.81 %。

新表述要点：
- 初位移 **84.37–97.46 %，SD 5.93**；30 Hz **71.77–87.91 %，SD 6.05**；
- 衰减率散布变大，但**受控残余幅值稳定在 1.0–9.0e-4 m**，散布主要来自分母——参数扰动使二阶模态从 29.65 Hz 偏到 28.1–31.3 Hz，无控制响应随之在 2.19–5.84e-3 之间变化；
- 结论从“参数不敏感”改为“残余振动水平不受参数扰动影响，衰减率的表观变化来自失谐而非控制失效”。

### 3.6 摘要（第 82 行）与结论（937、941 行）

| 行 | 现文 | 改为 | 依据 |
|---|---|---|---|
| 82 | by more than 80% | **by more than 77%** | MLS 延迟 77.77 |
| 937 | up to 83% under fixed-frequency | **up to 86%** | 30 Hz 延迟 86.08 |
| 941 | by more than 80% | **by more than 77%** | 同上 |

### 3.7 符号、公式与图

| 位置 | 处理 |
|---|---|
| 第 179 行符号表 | `T = L_h + L_f` → `T = \lfloor L_f/2 \rfloor + L_f` |
| 第 532、552 行公式 | 同上，`T={L}_{h}+{L}_{f}` → `T=\lfloor L_f/2\rfloor+L_f` |
| 第 469–472 行 | 补一句：只有最近 `L_f/2` 个隐状态与 `L_f` 个前向隐状态送入注意力 |
| 第 645 行（超参数表） | `$len_{history}$ & 36` **保留**（GRU 输入长度，代码事实） |
| 第 653 行 | “was set to 36” **保留**；其后补半句 “while only the most recent `L_f/2` hidden states are passed to the attention layer” |
| **fig6.pdf / fig7.pdf** | 图中 “State Seq **36**” → “State Seq **$L_h$**”，“Predicted Seq **4**” → “**$L_f$**”（两张都要改） |
| 第 472 行、Algorithm 1 | **不改**，意见① 已满足 |

> 图件是 PDF。文本替换最好在 Visio/Origin 源文件上改后重新导出；若 `论文/*.vsdx` 中没有这两张图，可用 PyMuPDF 覆盖补字，但字形会与原图不一致。

### 3.8 交叉引用

三张表现在自带统计量，**不再需要**指向新小节，原计划的三处交叉引用取消。

---

## 四、三项拍板结果

### A. `T = L_h + L_f` 的处理 —— 已定

- 正文与超参数表中的 **36 保留**：它是 GRU 输入序列长度，与代码一致（输入张量 `(1,36,5)`）；
- **公式改为 `T = \lfloor L_f/2 \rfloor + L_f`**：代码中注意力只用最近 `L_f/2 = 2` 个隐状态 + 4 个前向隐状态，合计 6；
- **图中 36 → `L_h`、4 → `L_f`**（唯一需要动图的地方）；
- **回复信**说明：时延环境下最近测量已陈旧，控制器更依赖未来状态的表征，故只取 `L_h = L_f/2` 个历史隐状态送入注意力。

### B. 为什么 No Control 会变 —— 两层原因，都正常

1. **组间（不同参数组）变化是物理必然**：±10% 的 `M`、`k_M` 把二阶模态从 29.65 Hz 推到 28.1–31.3 Hz，30 Hz 激励的放大倍数随之剧变，所以 30 Hz 的 NC 从 5.84e-3 掉到 2.19e-3（2.7 倍）；初位移工况变化小（4.51–4.90e-3，±4 %），因为自由衰减峰值主要由初始位移与阻尼比决定。
2. **组内（同参数组、10 个种子）变化是噪声实现差异**：每个种子对应一组不同的延迟序列与 `dt` 抖动。实测组内波动（`exp3b_summary.csv`）：

| 工况 | 组内 NC 的 SD | 相对波动 |
|---|---|---|
| 初位移（全部 5 组） | 1e-6 | **0.01–0.03 %** |
| 30 Hz 名义组 | 5.5e-4 | **9.4 %**（5.03–6.35e-3） |
| 30 Hz 其余 4 组 | 3e-6 – 2.2e-4 | 0.12–4.6 % |

即：**初位移工况的无控制响应与随机实现几乎无关**（自由衰减峰值由初始位移与阻尼比决定），而 **30 Hz 近共振工况对激励相位极敏感**，同样的 ±10% 参数偏移下 NC 会差 2.7 倍。这正是"每组 10 次配对 + 每组独立分母"的理由。

结论：表 7 的 NC 变化主体是**参数组不同**（物理必然），不是种子不同（初位移下几乎为零、30 Hz 下最多 9 %）。

### C. 预测诊断进正文的段落 —— 见第五节

---

## 五、预测诊断段落（可直接使用）

### 5.1 正文版（建议插在第 472 行那段之后，约 90 词，不写具体数字）

> **EN.** Two clarifications about this branch are necessary. First, it does not carry a calibrated physical prediction: no prediction loss, prediction target or pre-training stage is used, and the fully connected predictor is shaped solely by the actor and critic gradients of Eqs.~(\ref{eq:critic_loss}) and (\ref{eq:actor_obj}), so the rolled-forward quantities are a near-future latent context rather than an estimate of the measured response. Second, this is deliberate: under a total loop delay the most recent measurements are already stale, and what the controller needs is a representation that emphasises the anticipated evolution of the response rather than an accurate reconstruction of past samples. The attention pooling layer weights this context together with the observed history, and the whole branch is trained end-to-end with the control objective.
>
> **中文.** 关于这一分支需要两点澄清。第一，它并不输出经过标定的物理预测：整个分支没有预测损失、预测目标或预训练阶段，全连接预测器只由 critic 与 actor 目标的梯度塑造，因此前向展开得到的量是"近未来潜在上下文"，而不是对实测响应的估计。第二，这是有意为之：在总时延存在时，最近的测量已经陈旧，控制器真正需要的是强调响应预期演化的表征，而非对历史样本的精确重构。注意力池化层将该上下文与观测历史一起加权，整个分支随控制目标端到端训练。

### 5.2 回复信版（含实测数字）

> **EN.** We checked the branch empirically before revising the text. Feeding the true observation history of a trained controller (T4-2) into the network and comparing the rolled-forward outputs with the measured response gives a normalised RMSE of 1.00–1.02 at every prediction step and on every channel, i.e. no better than a persistence baseline (0.26–0.69 on the displacement channel); the outputs are O(1) in magnitude while the physical channels are of order 1e-3 m and 10 m/s², and the induced attention weights are 0.10–0.21 per step. We therefore do not claim any forecasting accuracy for this branch, and we have revised Section 4.2 accordingly: it is a latent, control-shaped representation, retained because it improves the closed-loop performance, not because it reproduces the measured state.
>
> **中文.** 修改文字之前我们对这一分支做了实测：把训练好的控制器（T4-2）的真实观测历史送入网络，再将前向展开的输出与实测响应比较，各预测步长、各通道的归一化 RMSE 均为 1.00–1.02，并不优于"保持上一时刻"的基线（位移通道 0.26–0.69）；输出量级为 O(1)，而物理通道分别为 1e-3 m 与 10 m/s² 量级，注意力权重为 0.10–0.21。因此我们不为该分支主张任何预测精度，并已相应修改 4.2 节：它是受控制目标塑造的潜在表征，保留它的理由是可改善闭环性能，而非复现实测状态。

---

## 六、执行顺序（确认后按此推进）

1. 替换 Table 3、Table 4、Table 5（`table_cond_*.tex`），去掉 Table 4 数据行的蓝色标记；
2. 替换 Table 7，行标签改为 `(0.9M,K,C)` 系列；
3. 4.4 开头插入指标说明 3 句；改 4.4.1（767 行）、4.4.2（774、776 行）、4.4.3（814 行）；
4. 重写 4.4.5 第 920 行；
5. 改摘要第 82 行、结论 937、941 行；
6. 改符号表 179 行与公式 532、552 行，补 469–472 行的 `L_f/2` 说明；
7. 插 5.1 的正文段落（第 472 行之后）；
8. 处理 fig6/fig7 的 36 → `L_h`（需源文件，见 3.7）；
9. 全量编译，检查未定义引用与 Overfull；
10. 撰英文逐点回复信（用 5.2 与第四节 B 的内容）。

import os
import json
import numpy as np
import torch
from collections import defaultdict
from typing import Any, Dict, Tuple, List, Optional
import matplotlib.pyplot as plt


class EpisodeRecorder:
    """灵活的单轮记录器，自动接受新字段。"""

    def __init__(self) -> None:
        self._data: Dict[str, List[Any]] = defaultdict(list) # 使用defaultdict自动初始化列表

    def append(self, **fields: Any) -> None:
        """添加一条记录，可以包含任意字段。"""
        for key, value in fields.items():
            self._data[key].append(value)

    def __len__(self) -> int:
        """返回记录的条目数。"""
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def as_numpy(self, keys=None) -> Dict[str, np.ndarray] | np.ndarray:
        """以 NumPy 数组形式返回记录数据。\n
        如果提供 keys 列表，则只返回指定字段。\n
        如果 keys 为 None，则返回所有字段。\n
        如果 keys 是字符串，则只返回该字段的数组。"""
        if keys is None:
            return {k: np.array(v) for k, v in self._data.items()}
        elif isinstance(keys, str):
            return np.array(self._data[keys])
        else:
            return {k: np.array(self._data[k]) for k in keys if k in self._data}
    
    def get_data(self) -> Dict[str, List[Any]]:
        """获取原始记录数据字典。"""
        return self._data

    def clear(self) -> None:
        """清空所有记录。"""
        self._data.clear()


class TrainingHistory:
    """Aggregates episode-level scalars; auto-expands when new metrics are added."""

    def __init__(self) -> None:
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self.current_episode: int = 1
        self.checkpoint_name: str = ""

    def log(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self._metrics[key].append(float(value))
        self.current_episode += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "current_episode": self.current_episode,
            "metrics": {k: np.array(v) for k, v in self._metrics.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainingHistory":
        obj = cls()
        obj.checkpoint_name = payload.get("checkpoint_name", "")
        obj.current_episode = int(payload.get("current_episode", 0))
        for key, arr in payload.get("metrics", {}).items():
            obj._metrics[key] = list(np.array(arr).tolist())
        return obj

    def metric_arrays(self) -> Dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self._metrics.items()}


def save_checkpoint(path: str, agent_state: Dict[str, Any], episode: EpisodeRecorder, history: TrainingHistory, extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "agent": agent_state,
        "episode": episode.as_numpy(),
        "history": history.to_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def latest_checkpoint(directory: str, suffix: str = ".pth") -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


## 通用的绘图函数模块
def plot_data(
    x_values_list: List[np.ndarray],
    y_values_list: List[np.ndarray],
    figsize: Tuple[int, int] = (16, 9),
    subplot: Optional[Tuple[int, int]] = None,
    plot_title: Optional[str] = None,
    subplot_titles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    line_styles: Optional[List[str]] = None,
    legends: Optional[List[str]] = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_grid: bool = False,
    log_scale: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    通用绘图：支持单图或子图栅格、多条曲线、自定义样式和保存。

    主要参数说明：
    - x_values_list / y_values_list: 支持列表或单个 ndarray，长度需一致；当 y_values_list 是嵌套列表时，表示“每个子图一组多条曲线”。
    - subplot: (rows, cols)。若提供则按顺序填充子图；不足的子图留空。
    - subplot_titles: 与子图数相同的标题列表（可选）。
    - colors/line_styles/legends: 可选的样式列表；若传入嵌套列表则按子图分别使用；否则对所有曲线复用。
    - log_scale: y 轴对数坐标。
    - save_path: 路径字符串；若提供则保存为 svg（文件名由 plot_title 或默认 plot.svg）。
    """

    # 规范化输入：支持 y_values_list 为 [curve1, curve2,...] 或 [[subplot1_curve...], [subplot2_curve...]]
    if not isinstance(y_values_list, list):
        y_values_list = [y_values_list]
    is_grouped = subplot is not None and any(isinstance(v, (list, tuple)) for v in y_values_list)

    # 处理 x 列表，允许同样的嵌套结构或单个共享 x
    if x_values_list is None:
        if is_grouped:
            x_values_list = [[np.arange(len(c)) for c in curves] for curves in y_values_list]
        else:
            x_values_list = [np.arange(len(y_values_list[i])) for i in range(len(y_values_list))]
    if not isinstance(x_values_list, list):
        x_values_list = [x_values_list]

    if is_grouped:
        # 子图数量 num_groups = len(y_values_list) 
        if subplot is None:
            raise ValueError("分组曲线绘制需要提供 subplot 布局")
    else:
        num_curves = len(y_values_list)
        if len(x_values_list) not in (1, num_curves):
            raise ValueError("x_values_list 长度必须为 1 或与 y_values_list 相同")

    # 布局：单图或子图
    if subplot:
        rows, cols = subplot
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes_flat = [ax]

    if plot_title:
        fig.suptitle(plot_title, fontsize=16)

    # 样式选择辅助
    def pick_style(arr, sub_idx, curve_idx):
        if arr is None:
            return None
        if arr and isinstance(arr[0], (list, tuple)):
            group = arr[sub_idx] if sub_idx < len(arr) else []
            return group[curve_idx] if curve_idx < len(group) else None
        return arr[curve_idx] if curve_idx < len(arr) else None

    # 绘制
    if is_grouped:
        for g_idx, curves in enumerate(y_values_list):
            ax = axes_flat[g_idx] if g_idx < len(axes_flat) else axes_flat[-1]
            x_group = x_values_list[g_idx] if g_idx < len(x_values_list) else x_values_list[0]
            if not isinstance(curves, (list, tuple)):
                curves = [curves]
            for c_idx, yv in enumerate(curves):
                xv = x_group[0] if isinstance(x_group, (list, tuple)) and len(x_group) == 1 else (x_group[c_idx] if isinstance(x_group, (list, tuple)) else x_group)
                if len(xv) != len(yv):
                    raise ValueError(f"子图 {g_idx} 曲线 {c_idx} 的 x 与 y 长度不匹配: {len(xv)} vs {len(yv)}")
                color = pick_style(colors, g_idx, c_idx)
                label = pick_style(legends, g_idx, c_idx)
                style = pick_style(line_styles, g_idx, c_idx) or "-"
                ax.plot(xv, yv, color=color, linestyle=style, label=label)

            if log_scale:
                ax.set_yscale("log")
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if show_grid:
                ax.grid(True)
            if show_legend and legends:
                ax.legend(loc=legend_loc, fontsize=12)
            if subplot and subplot_titles and g_idx < len(subplot_titles):
                ax.set_title(subplot_titles[g_idx])
    else:
        for idx, yv in enumerate(y_values_list):
            ax = axes_flat[idx] if idx < len(axes_flat) else axes_flat[-1]
            xv = x_values_list[0] if len(x_values_list) == 1 else x_values_list[idx]
            if len(xv) != len(yv):
                raise ValueError(f"曲线 {idx} 的 x 与 y 长度不匹配: {len(xv)} vs {len(yv)}")

            color = pick_style(colors, 0, idx)
            label = pick_style(legends, 0, idx)
            style = pick_style(line_styles, 0, idx) or "-"
            ax.plot(xv, yv, color=color, linestyle=style, label=label)

            if log_scale:
                ax.set_yscale("log")
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if show_grid:
                ax.grid(True)
            if show_legend and legends:
                ax.legend(loc=legend_loc, fontsize=12)
            if subplot and subplot_titles and idx < len(subplot_titles):
                ax.set_title(subplot_titles[idx])

    # 统一标签（仅对首个轴设置，子图可通过 subplot_titles 区分）
    if xlabel:
        for ax in axes_flat[-cols:] if subplot else [axes_flat[0]]:
            ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        for ax in axes_flat[::cols] if subplot else [axes_flat[0]]:
            ax.set_ylabel(ylabel, fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.96) if plot_title else None)

    # 保存和显示
    if save_path:
        fname = f"{plot_title}.svg" if plot_title else "plot.svg"
        plt.savefig(os.path.join(save_path, fname), format='svg')
    if show:
        plt.show()
    else:
        plt.close(fig)
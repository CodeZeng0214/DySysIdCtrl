import os
import json
import numpy as np
import torch
from collections import defaultdict
from typing import Any, Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import inspect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        """记录一组指标，自动扩展新指标。"""
        for key, value in metrics.items():
            self._metrics[key].append(float(value))
        self.current_episode += 1

    def to_dict(self) -> Dict[str, Any]:
        """导出训练历史记录为字典格式。"""
        return {
            "checkpoint_name": self.checkpoint_name,
            "current_episode": self.current_episode,
            "metrics": {k: np.array(v) for k, v in self._metrics.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainingHistory":
        """从字典加载训练历史记录。"""
        obj = cls()
        obj.checkpoint_name = payload.get("checkpoint_name", "")
        obj.current_episode = int(payload.get("current_episode", 0))
        for key, arr in payload.get("metrics", {}).items():
            obj._metrics[key] = list(np.array(arr).tolist())
        return obj

    def metric_arrays(self) -> Dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self._metrics.items()}
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """获取原始的指标数据字典。"""
        return self._metrics


def save_checkpoint(path: str, agent_state: Dict[str, Any], episode: EpisodeRecorder, history: TrainingHistory, extra: Optional[Dict[str, Any]] = None) -> None:
    """保存训练检查点，包括控制器状态、当前回合数据和训练历史记录。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "agent": agent_state,
        "episode": episode.as_numpy(),
        "history": history.to_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """加载训练检查点，返回包含控制器状态、当前回合数据和训练历史记录的字典。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    return payload


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def latest_checkpoint(directory: str, suffix: str = ".pth") -> Optional[str]:
    """获取目录下最新的检查点文件路径，若无则返回 None。"""
    if not os.path.isdir(directory):
        return None
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


## 通用的绘图函数模块
def plot_data(x_values: np.ndarray, y_values: np.ndarray, sub_group: Optional[List[Tuple]] = None,
              figsize: Tuple[int, int] = (16, 9), sub_shape: Optional[Tuple[int, int]] = None, 
              plot_title: Optional[str] = None, subplot_titles: Optional[List[str]] = None,
              colors: Optional[List[str]] = None, line_styles: Optional[List[str]] = None,
              legends: Optional[List[str]] = None, show_legend: bool = True, legend_loc: str = "best",
              xlabel: Optional[str] = None, ylabel: Optional[str] = None,
              xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None,
              show_grid: bool = False, log_scale: bool = False,
              save_path: Optional[str] = None, show: bool = True,
) -> None:
    """
    通用绘图：支持单图或子图栅格、多条曲线、自定义样式和保存。

    主要参数说明：
    - x_value / y_values: 支持单个 ndarray，y_values 可以是 1D 或 2D；当 y_values 是 2D 时，表示多条曲线。
    - sub_group: 分组信息，若提供则按组绘制子图，每组内多条曲线。
    - sub_shape: (rows, cols) 子图布局，若未提供则自动计算。
    - subplot_titles: 与子图数相同的标题列表（可选）。
    - colors/line_styles/legends: 可选的样式列表；若传入嵌套列表则按子图分别使用；否则对所有曲线复用。
    - log_scale: y 轴对数坐标。
    - save_path: 路径字符串；若提供则保存为 svg（文件名由 plot_title 或默认 plot.svg）。
    """
    # 设置中文字体和GPU
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 规范化 y 输入：接受 ndarray（1D/2D）或列表，统一转换为列表结构
    if isinstance(y_values, np.ndarray):
        if y_values.ndim == 1:
            y_values = y_values.reshape(-1, 1)
        elif y_values.ndim == 2:
            pass
        else:
            raise ValueError("y_values ndarray 仅支持 1D 或 2D")
    elif isinstance(y_values, list):
        y_values = np.array(y_values).reshape(-1, y_values[0].__len__())

    # 规范化 y 输入：接受 ndarray（1D/2D）或列表，统一转换为列表结构
    if x_values is None: # 自动生成 x_value
        x_values = np.tile(np.arange(y_values.shape[0]), y_values.shape[1])
    if isinstance(x_values, np.ndarray):
        if x_values.ndim == 1:
            x_values = x_values.reshape(-1, 1)
        elif x_values.ndim == 2:
            pass
        else:
            raise ValueError("x_values ndarray 仅支持 1D 或 2D")
    elif isinstance(x_values, list):
        x_values = np.array(x_values).reshape(-1, x_values[0].__len__())

    # 布局：单图或子图
    num_subplots = len(sub_group)
    if sub_shape:
        rows, cols = sub_shape
    else:
        rows, cols = 1, num_subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    if plot_title:
        fig.suptitle(plot_title, fontsize=16)

    # 样式选择辅助
    def pick_style(arr, curve_idx):
        if arr is None:
            return None
        return arr[curve_idx] if curve_idx < len(arr) else None

    # 绘制：按分组在子图中绘制多条曲线
    for g_idx, grp in enumerate(sub_group):
        ax = axes_flat[g_idx] if g_idx < len(axes_flat) else axes_flat[-1]
        for idx in grp:
            if idx >= y_values.shape[1]:
                raise IndexError(f"sub_group 索引 {idx} 超出 y_values 列数 {y_values.shape[1]}")
            xv = x_values[:, idx]
            yv = y_values[:, idx]
            if xv.shape[0] != yv.shape[0]:
                raise ValueError(f"子图 {g_idx} 曲线 {idx} 的 x 与 y 长度不匹配: {len(xv)} vs {len(yv)}")
            color = pick_style(colors, idx)
            label = pick_style(legends, idx)
            style = pick_style(line_styles, idx) or "-"
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
        if subplot_titles and g_idx < len(subplot_titles):
            ax.set_title(subplot_titles[g_idx])

    # 统一标签（仅对首个轴设置，子图可通过 subplot_titles 区分）
    if xlabel:
        for ax in axes_flat[-cols:]:
            ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        for ax in axes_flat[::cols]:
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

def format_special_type(obj):
    """
    将特殊类型（numpy数组、函数）转换为可读字符串，其他类型保持原样。
    """
    # 处理numpy数组
    if isinstance(obj, np.ndarray):
        # 转为列表后，标注为numpy.ndarray类型（保留形状信息）
        return f"numpy.ndarray(shape={obj.shape}, dtype={obj.dtype}, data={obj.tolist()})"
    elif isinstance(obj, list):
        return str([format_special_type(item) for item in obj])
    # 处理函数（包括自定义函数、lambda、partial等）
    elif callable(obj):
        # 1. 普通函数/类方法：提取函数名
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            func_name = obj.__name__
            try:
                source = inspect.getsource(obj)
                label = "Lambda" if func_name == "<lambda>" else "Function"
                return f"{label}: \n    ```python\n     {source}\n```    "
            except (inspect.InspectError, OSError):
                return "Lambda function" if func_name == "<lambda>" else f"Function: {func_name}"
        # 3. 其他可调用对象（如partial）：返回repr
        else:
            return repr(obj)
    
    # 处理其他类型（如int、str、list等）
    else:
        return obj

def format_dict(input_dict)-> str:
    """
    递归处理字典中的所有值，转换特殊类型后返回原生字符串。
    """
    formatted_dict = {}
    for key, value in input_dict.items():
        # 若值是字典，递归处理
        if isinstance(value, dict):
            formatted_dict[key] = format_dict(value)
        else:
            formatted_dict[key] = format_special_type(value)
    return formatted_dict

def make_dirs(project_name: str) -> Tuple[str, str, str]:
    """确保目录存在，若不存在则创建。"""
    # 创建项目主目录
    project_path = os.path.join("savedata", project_name)
    os.makedirs(project_path, exist_ok=True)

    # 创建检查点保存目录
    save_checkpoint_path = os.path.join(project_path, "checkpoints")
    os.makedirs(save_checkpoint_path, exist_ok=True)

    # 创建绘图保存目录
    save_plot_path = os.path.join(project_path, "plots")
    os.makedirs(save_plot_path, exist_ok=True)

    return project_path, save_checkpoint_path, save_plot_path
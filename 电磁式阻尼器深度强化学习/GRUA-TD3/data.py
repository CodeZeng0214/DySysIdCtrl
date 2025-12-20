import os
import json
import numpy as np
import torch
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


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

    def as_numpy(self) -> Dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self._data.items()}

    def clear(self) -> None:
        """清空所有记录。"""
        self._data.clear()


class TrainingHistory:
    """Aggregates episode-level scalars; auto-expands when new metrics are added."""

    def __init__(self) -> None:
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self.current_episode: int = 0
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

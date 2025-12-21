import numpy as np
from typing import Callable


ACTION_BOUND = 5.0


def zero(t: float) -> float:
    return 0.0


def sin_wave(amplitude: float = 0.01, frequency: float = 1.0, phase: float = 0.0) -> Callable[[float], float]:
    """生成一个正弦波函数。\n
     参数：
        amplitude: 振幅
        frequency: 频率（Hz）
        phase: 相位（弧度）"""
    def func(t: float) -> float:
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return func


def tolerance_reward(tolerance: float = 1e-3) -> Callable[[np.ndarray, float, np.ndarray], float]:
    """生成一个基于容差的奖励函数。\n
    参数：
        tolerance: 容差范围"""
    def reward_fn(state: np.ndarray, action: float, next_state: np.ndarray) -> float:
        x2 = state[3] # 主结构位移
        next_x2 = next_state[3] # 下一步主结构位移
        reward = 0.0
        if abs(next_x2) <= tolerance:
            reward += (tolerance - abs(next_x2)) / tolerance
            if abs(next_x2) <= abs(x2):
                reward += 1.0
        else:
            reward -= 1.0
            if abs(next_x2) > abs(x2):
                reward -= 1.0
            reward += -np.log10(abs(next_x2) / tolerance)
        reward -= abs(action) / ACTION_BOUND
        return float(np.clip(reward / 4.0, -1.0, 1.0))
    return reward_fn

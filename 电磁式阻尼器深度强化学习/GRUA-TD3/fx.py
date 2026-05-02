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
    def tolerance_rewardfx(state: np.ndarray, action: float, next_state: np.ndarray) -> float:
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
        reward -= abs(action) / ACTION_BOUND / 2
        return float(np.clip(reward / 4.0, -1.0, 1.0))
    return tolerance_rewardfx

def tolerance_smooth_reward(tolerance: float = 1e-3) -> Callable[[np.ndarray, float, np.ndarray], float]:
    """生成一个基于容差的平滑奖励函数。\n
    参数：
        tolerance: 容差范围"""
    def tolerance_rewardfx(state: np.ndarray, action: float, next_state: np.ndarray) -> float:
        x2 = state[3] # 主结构位移
        next_x2 = next_state[3] # 下一步主结构位移
        reward = 0.0
        if abs(next_x2) <= tolerance:
            reward += (tolerance - abs(next_x2)) / tolerance # 随着位移接近tolerance线性增加，最大1分
            if abs(next_x2) <= abs(x2):
                reward += (abs(x2) - abs(next_x2)) / (abs(x2) + 1e-14) # 如果位移变小，额外奖励，随着位移减小的幅度线性增加，最大1分
        else:
            reward -= 2 * np.log10(abs(next_x2) / tolerance) # 超出容差范围的部分，按照位移大小的对数惩罚，越大惩罚越重，由clip决定不会超过-3分
            if abs(next_x2) > abs(x2):
                reward -= (abs(next_x2) - abs(x2)) / (abs(next_x2) + 1e-14) # 如果位移变大，惩罚，随着位移增大的幅度线性增加，最大-1分
        reward -= abs(action) / ACTION_BOUND / 4 # 控制输入的惩罚，最大-0.25分
        return float(np.clip(reward / 4.0, -1.0, 1.0))
    return tolerance_rewardfx

def mls_force(amplitude: float = 0.01, frequency: float = 10.0, phase: float = 0.0) -> Callable[[float], float]:
    """生成一个基于MLS(伪随机二激码)信号的激励函数。\n
    参数：
        amplitude: 振幅
        frequency: 状态切换频率 (即MLS时钟频率，决定了信号的带宽，Hz)
        phase: 时间相位偏移（秒）
    """
    from scipy.signal import max_len_seq
    # 生成一个较长的MLS序列，14阶对应的长度为 2^14 - 1 = 16383，足够覆盖一般的仿真时长
    mls_seq, _ = max_len_seq(14)
    mls_seq = mls_seq * 2.0 - 1.0  # 变换为1和-1的序列
    seq_len = len(mls_seq)
    
    def func(t: float) -> float:
        # 将时间t依据frequency转换为索引，模拟时钟节拍
        active_t = max(0.0, t + phase)
        index = int(active_t * frequency) % seq_len
        return amplitude * mls_seq[index]
    return func

import numpy as np
from typing import Callable
from scipy.linalg import expm
action_bound = 5

def enhanced_reward_func(obs:np.ndarray, action:np.ndarray, next_obs:np.ndarray)-> float:
    """增强型奖励函数"""
    x1, v1, a1 = obs[:3]
    x2, v2, a2 = obs[3:6]
    next_x1, next_v1, next_a1 = next_obs[:3]
    next_x2, next_v2, next_a2 = next_obs[3:6]
    
    # 位移和速度改善奖励
    position_improvement = 10.0 * (abs(x2) - abs(next_x2))
    velocity_improvement = 5.0 * (abs(v2) - abs(next_v2))
    
    # 位移和速度幅值惩罚
    position_penalty = -2.0 * abs(next_x2)
    velocity_penalty = -1.0 * abs(next_v2)
    
    # 动作变化惩罚 (鼓励平滑的控制信号)
    action_smoothness = -0.5 * np.sum(action**2)
    
    # 鼓励相对于吸振器的反相位运动
    phase_reward = 0.0
    if np.sign(next_x1) != np.sign(next_x2):
        phase_reward = 0.5
    
    return position_improvement + velocity_improvement + position_penalty + velocity_penalty + action_smoothness + phase_reward

def tolerance_if_rf(tolerance: float = 1e-3) -> Callable:
    """基于容忍度的奖励函数\n
    判断当前状态和下一个状态是否在容忍范围内，并根据情况给予奖励或惩罚\n
    仅使用了平台的位移作为奖励基准\n"""
    def reward_func(obs:np.ndarray, action:np.ndarray, next_obs:np.ndarray)-> float:
        obs = obs.tolist()
        next_obs = next_obs.tolist() # 将张量转换为列表
        x1, v1, a1 = obs[:3]
        x2, v2, a2 = obs[3:6]
        next_x1, next_v1, next_a1 = next_obs[:3]
        next_x2, next_v2, next_a2 = next_obs[3:6]
        
        reward = 0
        if (abs(next_x2) <= tolerance):
            # 处于容忍范围内的线性奖励
            reward += ((tolerance-abs(next_x2)) / tolerance)
            if (abs(next_x2) <= abs(x2)):
            # 处于容忍范围内且下一个状态更接近容忍范围
                reward += 1.0

        elif (abs(next_x2) > tolerance):
            # 处于容忍范围外
            reward += -1.0
            if (abs(next_x2) > abs(x2)):
            # 处于容忍范围外且下一个状态更远离容忍范围
                reward += -1.0
            # 基于nx2比tol的值的数量级增大惩罚
            reward += (-np.log10(abs(next_x2) / tolerance))
        else:
            reward += 0
            
        # 动作过大的惩罚
        reward += -(abs(float(action)) / action_bound)
        
        return reward
    return reward_func

def tolerance_exp_rf(tolerance: float = 1e-3) -> Callable:
    """基于容忍度的指数奖励函数\n
    适用于位移的奖励函数"""
    def reward_func(obs:np.ndarray, action:np.ndarray, next_obs:np.ndarray) -> float:
        """基于容忍度的奖励函数\n
        """
        obs = obs.tolist()
        next_obs = next_obs.tolist() # 将张量转换为列表
        x1, v1, a1 = obs[:3]
        x2, v2, a2 = obs[3:6]
        next_x1, next_v1, next_a1 = next_obs[:3]
        next_x2, next_v2, next_a2 = next_obs[3:6]
        
        reward = expm(-abs(next_x2) / tolerance) - 1.0
        return reward.reshape(-1)
    return reward_func


## 定义扰动函数
def zero(t):
    """零扰动"""
    return 0

def sin_wave(amplitude=0.01, frequency=1.0, phase=0):
    """正弦波扰动"""
    def func(t):
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return func
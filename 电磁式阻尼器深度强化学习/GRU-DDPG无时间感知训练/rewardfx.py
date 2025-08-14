import numpy as np
from typing import Callable
from scipy.linalg import expm

def test_reward_function(obs, action, next_obs):
    """测试奖励函数"""
    x1, v1, a1 = obs[:3]
    x2, v2, a2 = obs[3:6]
    next_x1, next_v1, next_a1 = next_obs[:3]
    next_x2, next_v2, next_a2 = next_obs[3:6]
    
    # 奖励函数
    reward = - (1000 * abs(float(next_x2)) + 0.1 * abs(float(action)))
    
    return float(reward)

def better_reward_function(obs, action, next_obs):
    """提供更多梯度信息的奖励函数"""
    x1, v1, a1 = obs[:3]
    x2, v2, a2 = obs[3:6]
    next_x1, next_v1, next_a1 = next_obs[:3]
    next_x2, next_v2, next_a2 = next_obs[3:6]
    
    # 减振效果奖励: 状态减小给予正奖励
    state_improvement = 0 # 2.0 * (abs(x2) - abs(next_x2))
    
    # 状态值越小越好
    position_reward = -5.0 * abs(next_x2)
    
    # 动作平滑度奖励
    action_smoothness = 0 # -0.05 * action**2
    
    # 相位差异奖励: 鼓励主结构和吸振器反相运动
    phase_reward = 0.0
    # if np.sign(next_v1) != np.sign(next_v2):
    #     phase_reward = 0.2 * abs(next_v1 - next_v2)
        
    total_reward = state_improvement + position_reward + action_smoothness + phase_reward
    return float(total_reward)

def squared_reward_function(obs, action, next_obs):
    """平方奖励函数"""
    x1, v1, a1 = obs[:3]
    x2, v2, a2 = obs[3:6]
    next_x1, next_v1, next_a1 = next_obs[:3]
    next_x2, next_v2, next_a2 = next_obs[3:6]
    
    reward = - (10 * (1e3 * next_x2)**2 + 1.0 * float(action)**2)
    return float(reward)


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
            # 处于容忍范围内
            reward += 0.5
            if (abs(next_x2) <= abs(x2)):
            # 处于容忍范围内且下一个状态更接近容忍范围
                reward += 1.0
        elif (abs(next_x2) > tolerance):
            # 处于容忍范围外
            reward += -0.5
            if (abs(next_x2) > abs(x2)):
            # 处于容忍范围外且下一个状态更远离容忍范围
                reward += -1.0
        else:
            reward += 0
        return reward
    return reward_func

def tolerance_liner_rf(tolerance: float = 1e-3) -> Callable:
    """基于容忍度的线性奖励函数\n"""
    def reward_func(obs:np.ndarray, action:np.ndarray, next_obs:np.ndarray) -> float:
        """基于容忍度的奖励函数\n
        """
        obs = obs.tolist()
        next_obs = next_obs.tolist() # 将张量转换为列表
        x1, v1, a1 = obs[:3]
        x2, v2, a2 = obs[3:6]
        next_x1, next_v1, next_a1 = next_obs[:3]
        next_x2, next_v2, next_a2 = next_obs[3:6]
        
        if abs(tolerance) >= abs(next_x2):
            # 处于容忍范围内
            reward = (abs(tolerance) - abs(next_x2)) / tolerance
        else:
            # 处于容忍范围外
            reward = 1.0 * (abs(tolerance) - abs(next_x2)) / tolerance
            if abs(reward) > 10:
                reward = -10.0

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


def exp_reward_func(tolerance=None, select_dim=None)-> float:
    obs = obs[select_dim]
    next_obs = next_obs[select_dim]
    def reward_func(obs:np.ndarray, action:np.ndarray, next_obs:np.ndarray):
        
        sum_next_obs = np.sum(next_obs**2) # 计算下一个状态的平方和
        sum_action = np.sum(action**2) # 计算动作的平方和
        reward = -0.5 * (sum_next_obs + sum_action) # 奖励函数
    return reward_func()


def x_reward_func(obs:np.ndarray, action:np.ndarray, next_obs:np.ndarray, tolerance)-> float:
    """适用于位移的奖励函数"""
    obs = obs.item()
    next_obs = next_obs.item()
    if abs(obs) < tolerance and abs(next_obs) < abs(obs):
        # 处于容忍范围内或下一个状态更接近容忍范围
        reward = abs(tolerance - abs(next_obs) / tolerance)
    else:
        # 处于容忍范围外或下一个状态更远离容忍范围
        if (abs(next_obs) > tolerance) and (abs(next_obs) > abs(obs)):
            reward = -0.5 
        else:
            reward = 0
    return reward


import numpy as np
from typing import List, Union, Callable
from functools import wraps

def select_dimensions(dimensions: Union[int, List[int]] = None):
    """
    装饰器: 选择观测维度进行奖励计算
    
    参数:
        dimensions: 要使用的观测维度索引，可以是单个索引或索引列表。None表示使用所有维度
    
    返回:
        修饰后的奖励函数，它将从obs和next_obs中提取指定维度进行计算
    """
    def decorator(reward_func: Callable):
        @wraps(reward_func)
        def wrapper(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, **kwargs):
            # 如果指定了维度，则只选择这些维度
            if dimensions is not None:
                if isinstance(dimensions, int):
                    # 单维度选择
                    selected_obs = np.array([obs[dimensions]])
                    selected_next_obs = np.array([next_obs[dimensions]])
                else:
                    # 多维度选择
                    selected_obs = np.array([obs[i] for i in dimensions])
                    selected_next_obs = np.array([next_obs[i] for i in dimensions])
                
                # 使用选择的维度调用原始奖励函数
                return reward_func(selected_obs, action, selected_next_obs)
            
            # 如果没有指定维度，使用所有维度
            return reward_func(obs, action, next_obs)
        
        return wrapper
    
    return decorator


# 示例1: 基于加速度的奖励函数
@select_dimensions(dimensions=[1, 3])  # 选择吸振器加速度和主结构加速度
def acceleration_reward(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, tolerance: float = 1e-3):
    """基于加速度的奖励函数"""
    sum_next_obs = np.sum(next_obs**2)  # 计算下一个状态的平方和
    sum_action = np.sum(action**2)  # 计算动作的平方和
    
    # 奖励函数: 惩罚大加速度和大动作
    reward = -0.5 * (sum_next_obs + 0.01 * sum_action)
    return reward


# 示例2: 基于位移的奖励函数
@select_dimensions(dimensions=2)  # 选择主结构位移
def displacement_reward(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, tolerance: float = 1e-3):
    """基于位移的奖励函数"""
    obs_val = obs.item()
    next_obs_val = next_obs.item()
    
    if abs(next_obs_val) < tolerance:
        # 位移在容差范围内，给予正奖励
        reward = 1.0 - (abs(next_obs_val) / tolerance)
    else:
        # 位移超出容差范围，给予负奖励
        reward = -0.5 * (abs(next_obs_val) / tolerance)
    
    # 对大动作施加惩罚
    action_penalty = -0.01 * np.sum(action**2)
    return reward + action_penalty


# 示例3: 自定义灵活的奖励函数
def create_custom_reward(dimensions=None):
    """创建自定义奖励函数"""
    @select_dimensions(dimensions=dimensions)
    def custom_reward(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, tolerance: float = 1e-3):
        # 实现自定义奖励逻辑
        obs_magnitude = np.linalg.norm(obs)
        next_obs_magnitude = np.linalg.norm(next_obs)
        
        # 奖励降低观测幅值
        if next_obs_magnitude < obs_magnitude:
            reward = 0.1 * (obs_magnitude - next_obs_magnitude) / obs_magnitude
        else:
            reward = -0.2 * (next_obs_magnitude - obs_magnitude) / obs_magnitude
        
        # 惩罚大动作
        action_penalty = -0.01 * np.sum(action**2)
        
        return reward + action_penalty
    
    return custom_reward


# 使用自定义奖励函数的示例
# position_velocity_reward = create_custom_reward(dimensions=[0, 1])  # 使用位移和速度
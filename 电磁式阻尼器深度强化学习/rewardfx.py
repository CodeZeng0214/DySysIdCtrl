import numpy as np

def a_reward_func(tolerance=None, select_dim=None)-> float:
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
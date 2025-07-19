#!/usr/bin/env python3
"""
时间感知GRU-DDPG使用示例
演示如何使用新的时间感知功能
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 导入必要的模块
from env import ElectromagneticDamperEnv
from ddpg_agent import GruDDPGAgent
from my_nn import Gru_ReplayBuffer

def create_simple_system():
    """创建简化的二自由度系统"""
    # 简化的系统参数
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-1875.0, -0.625, 1875.0, 0.625],
        [0.0, 0.0, 0.0, 1.0],
        [18.75, 0.05625, -393.75, -5.05625]
    ])
    B = np.array([[0.0], [28.125], [0.0], [-0.45]])
    C = np.array([[-1875.0, -0.625, 1875.0, 0.625], [18.75, 0.05625, -393.75, -5.05625]])
    D = np.array([[28.125], [-0.45]])
    E = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [200.0, 0.05]])
    
    return A, B, C, D, E

def demo_time_aware_training():
    """演示时间感知训练"""
    print("=== 时间感知GRU-DDPG训练演示 ===")
    
    # 系统参数
    A, B, C, D, E = create_simple_system()
    
    # 创建时间感知环境
    env = ElectromagneticDamperEnv(
        A, B, C, D, E, 
        Ts=0.001, T=2.0,  # 短时间仿真用于演示
        use_time_noise=True,
        time_noise_std=0.0001,
        obs_indices=[3, 5]  # 观测平台位移和加速度
    )
    
    # 创建时间感知智能体
    agent = GruDDPGAgent(
        state_dim=2,  # 观测维度
        action_dim=1,
        seq_len=10,
        hidden_dim=32,  # 较小的网络用于演示
        use_time_input=True  # 启用时间感知
    )
    
    # 创建时间感知回放池
    replay_buffer = Gru_ReplayBuffer(
        capacity=1000,
        batch_size=16,
        seq_len=10,
        use_time_input=True
    )
    
    print(f"环境配置: 时间噪声={env.use_time_noise}, 噪声标准差={env.time_noise_std}")
    print(f"智能体配置: 时间输入={agent.use_time_input}, 序列长度={agent.seq_len}")
    
    # 训练几个episode
    episode_rewards = []
    
    for episode in range(3):  # 只训练3个episode用于演示
        env.reset()
        agent.reset_state_history()
        replay_buffer.reset_history()
        
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 100:  # 限制步数
            obs = env.get_observation()
            dt = getattr(env, 'current_dt', env.Ts)
            
            # 选择动作（包含时间信息）
            action = agent.select_action(obs, add_noise=True, dt=dt)
            
            # 执行动作
            next_obs, reward, done, actual_dt = env.step(action)
            
            # 存储经验（包含时间信息）
            replay_buffer.add(obs, action, reward, next_obs, done, actual_dt)
            
            episode_reward += reward
            step_count += 1
            
            # 更新网络
            if len(replay_buffer) > 16:
                try:
                    critic_loss, actor_loss = agent.update(replay_buffer)
                except:
                    pass  # 忽略演示中的错误
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: 奖励={episode_reward:.2f}, 步数={step_count}")
    
    return env, agent, episode_rewards

def demo_time_comparison():
    """演示时间感知 vs 传统方法的对比"""
    print("\n=== 时间感知 vs 传统方法对比 ===")
    
    A, B, C, D, E = create_simple_system()
    
    # 创建扰动函数
    def sine_disturbance(t):
        return 0.005 * np.sin(2 * np.pi * 5 * t)  # 5Hz正弦扰动
    
    # 时间感知环境和智能体
    env_time = ElectromagneticDamperEnv(
        A, B, C, D, E, Ts=0.001, T=3.0,
        z_func=sine_disturbance,
        use_time_noise=True, time_noise_std=0.0002,
        obs_indices=[3]
    )
    
    agent_time = GruDDPGAgent(
        state_dim=1, action_dim=1, seq_len=10, 
        hidden_dim=32, use_time_input=True
    )
    
    # 传统环境和智能体
    env_normal = ElectromagneticDamperEnv(
        A, B, C, D, E, Ts=0.001, T=3.0,
        z_func=sine_disturbance,
        use_time_noise=False,
        obs_indices=[3]
    )
    
    agent_normal = GruDDPGAgent(
        state_dim=1, action_dim=1, seq_len=10, 
        hidden_dim=32, use_time_input=False
    )
    
    # 运行仿真
    print("运行时间感知仿真...")
    results_time = env_time.run_simulation(controller=agent_time)
    
    print("运行传统仿真...")
    results_normal = env_normal.run_simulation(controller=agent_normal)
    
    # 简单的性能对比
    time_rms = np.sqrt(np.mean(results_time['all_states'][:, 3]**2))
    normal_rms = np.sqrt(np.mean(results_normal['all_states'][:, 3]**2))
    
    print(f"时间感知GRU-DDPG位移RMS: {time_rms:.6f}")
    print(f"传统GRU-DDPG位移RMS: {normal_rms:.6f}")
    
    if 'dt_history' in results_time:
        dt_array = np.array(results_time['dt_history'])
        print(f"时间步长变化: 均值={np.mean(dt_array):.6f}, 标准差={np.std(dt_array):.6f}")
    
    return results_time, results_normal

def main():
    """主函数"""
    print("时间感知GRU-DDPG使用示例")
    print("=" * 50)
    
    # 演示1: 训练过程
    env, agent, rewards = demo_time_aware_training()
    
    # 演示2: 性能对比
    results_time, results_normal = demo_time_comparison()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("主要特点:")
    print("1. 时间感知网络能够处理变化的采样间隔")
    print("2. 训练过程中考虑时间噪声提高鲁棒性")
    print("3. 与传统方法保持完全兼容")
    print("\n使用方法:")
    print("- 设置 use_time_input=True 启用时间感知")
    print("- 设置 use_time_noise=True 启用时间噪声训练")
    print("- 在 select_action() 中传递 dt 参数")

if __name__ == "__main__":
    main()

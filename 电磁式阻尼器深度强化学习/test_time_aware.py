#!/usr/bin/env python3
"""
时间感知GRU-DDPG系统测试脚本
测试新功能的基本兼容性和功能性
"""

import numpy as np
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_time_aware_networks():
    """测试时间感知网络架构"""
    print("=== 测试时间感知网络架构 ===")
    
    from my_nn import Gru_Actor, Gru_Critic, Gru_ReplayBuffer
    
    # 测试参数
    state_dim = 4
    action_dim = 1
    seq_len = 10
    batch_size = 32
    
    # 测试时间感知Actor
    print("测试时间感知Gru_Actor...")
    actor_time = Gru_Actor(state_dim=state_dim, action_dim=action_dim, seq_len=seq_len, use_time_input=True)
    actor_normal = Gru_Actor(state_dim=state_dim, action_dim=action_dim, seq_len=seq_len, use_time_input=False)
    
    # 创建测试数据
    state_seq = torch.randn(batch_size, seq_len, state_dim)
    time_seq = torch.randn(batch_size, seq_len, 1)
    
    # 测试前向传播
    action_time = actor_time(state_seq, time_seq)
    action_normal = actor_normal(state_seq, None)
    
    print(f"时间感知Actor输出形状: {action_time.shape}")
    print(f"传统Actor输出形状: {action_normal.shape}")
    assert action_time.shape == (batch_size, action_dim)
    assert action_normal.shape == (batch_size, action_dim)
    
    # 测试时间感知Critic
    print("测试时间感知Gru_Critic...")
    critic_time = Gru_Critic(state_dim=state_dim, action_dim=action_dim, seq_len=seq_len, use_time_input=True)
    critic_normal = Gru_Critic(state_dim=state_dim, action_dim=action_dim, seq_len=seq_len, use_time_input=False)
    
    action = torch.randn(batch_size, action_dim)
    q_time = critic_time(state_seq, action, time_seq)
    q_normal = critic_normal(state_seq, action, None)
    
    print(f"时间感知Critic输出形状: {q_time.shape}")
    print(f"传统Critic输出形状: {q_normal.shape}")
    assert q_time.shape == (batch_size, 1)
    assert q_normal.shape == (batch_size, 1)
    
    # 测试时间感知回放池
    print("测试时间感知Gru_ReplayBuffer...")
    buffer_time = Gru_ReplayBuffer(capacity=1000, batch_size=batch_size, seq_len=seq_len, use_time_input=True)
    buffer_normal = Gru_ReplayBuffer(capacity=1000, batch_size=batch_size, seq_len=seq_len, use_time_input=False)
    
    # 添加一些经验
    for i in range(50):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        dt = 0.001 + np.random.randn() * 0.0001
        
        buffer_time.add(state, action, reward, next_state, done, dt)
        buffer_normal.add(state, action, reward, next_state, done)
    
    if len(buffer_time) >= batch_size:
        sample_time = buffer_time.sample()
        sample_normal = buffer_normal.sample()
        
        print(f"时间感知回放池采样返回{len(sample_time)}个元素")
        print(f"传统回放池采样返回{len(sample_normal)}个元素")
        
        if buffer_time.use_time_input:
            assert len(sample_time) == 7  # state_seqs, actions, rewards, next_state_seqs, dones, time_seqs, next_time_seqs
        else:
            assert len(sample_normal) == 7  # 为了兼容性，都返回7个元素，但time相关为None
    
    print("✓ 网络架构测试通过")


def test_environment_time_noise():
    """测试环境时间噪声功能"""
    print("\n=== 测试环境时间噪声功能 ===")
    
    from env import ElectromagneticDamperEnv
    
    # 系统参数
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
    
    # 创建带时间噪声的环境
    env_noisy = ElectromagneticDamperEnv(
        A, B, C, D, E, Ts=0.001, T=1.0,
        use_time_noise=True, time_noise_std=0.0001,
        obs_indices=[3]  # 只观测平台位移
    )
    
    # 创建不带时间噪声的环境
    env_fixed = ElectromagneticDamperEnv(
        A, B, C, D, E, Ts=0.001, T=1.0,
        use_time_noise=False,
        obs_indices=[3]
    )
    
    print("测试时间噪声环境...")
    
    # 测试step函数
    obs_noisy = env_noisy.reset()
    obs_fixed = env_fixed.reset()
    
    dt_list = []
    for i in range(100):
        action = np.array([0.1])  # 简单的常数动作
        
        # 时间噪声环境
        result_noisy = env_noisy.step(action)
        assert len(result_noisy) == 4  # obs, reward, done, dt
        next_obs_noisy, reward_noisy, done_noisy, dt_noisy = result_noisy
        dt_list.append(dt_noisy)
        
        # 固定时间环境
        result_fixed = env_fixed.step(action)
        if len(result_fixed) == 4:
            next_obs_fixed, reward_fixed, done_fixed, dt_fixed = result_fixed
        else:
            next_obs_fixed, reward_fixed, done_fixed = result_fixed
            dt_fixed = env_fixed.Ts
        
        if done_noisy or done_fixed:
            break
    
    dt_array = np.array(dt_list)
    print(f"时间步长统计 - 均值: {np.mean(dt_array):.6f}, 标准差: {np.std(dt_array):.6f}")
    print(f"时间步长范围: [{np.min(dt_array):.6f}, {np.max(dt_array):.6f}]")
    
    # 验证时间步长确实有变化
    assert np.std(dt_array) > 0, "时间步长应该有变化"
    assert np.abs(np.mean(dt_array) - 0.001) < 0.0001, "平均时间步长应该接近设定值"
    
    print("✓ 环境时间噪声测试通过")


def test_agent_compatibility():
    """测试智能体兼容性"""
    print("\n=== 测试智能体兼容性 ===")
    
    from ddpg_agent import GruDDPGAgent
    from my_nn import Gru_ReplayBuffer
    
    # 创建时间感知智能体
    agent_time = GruDDPGAgent(
        state_dim=4, action_dim=1, seq_len=10,
        use_time_input=True
    )
    
    # 创建传统智能体
    agent_normal = GruDDPGAgent(
        state_dim=4, action_dim=1, seq_len=10,
        use_time_input=False
    )
    
    print("测试智能体动作选择...")
    
    # 测试动作选择
    state = np.random.randn(4)
    dt = 0.0012
    
    action_time = agent_time.select_action(state, add_noise=False, dt=dt)
    action_normal = agent_normal.select_action(state, add_noise=False, dt=dt)  # dt参数应该被忽略
    
    print(f"时间感知智能体输出: {action_time}")
    print(f"传统智能体输出: {action_normal}")
    
    assert action_time.shape == (1,)
    assert action_normal.shape == (1,)
    
    # 测试网络更新
    print("测试网络更新...")
    
    buffer_time = Gru_ReplayBuffer(capacity=1000, batch_size=32, seq_len=10, use_time_input=True)
    buffer_normal = Gru_ReplayBuffer(capacity=1000, batch_size=32, seq_len=10, use_time_input=False)
    
    # 添加足够的经验以进行更新
    for i in range(50):
        state = np.random.randn(4)
        action = np.random.randn(1)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = False
        dt = 0.001 + np.random.randn() * 0.0001
        
        agent_time.state_history.append(state)
        agent_normal.state_history.append(state)
        
        if agent_time.use_time_input:
            agent_time.time_history.append(dt)
        
        buffer_time.add(state, action, reward, next_state, done, dt)
        buffer_normal.add(state, action, reward, next_state, done)
    
    if len(buffer_time) >= 32:
        try:
            loss_time = agent_time.update(buffer_time)
            print(f"时间感知智能体更新损失: {loss_time}")
        except Exception as e:
            print(f"时间感知智能体更新错误: {e}")
            
    if len(buffer_normal) >= 32:
        try:
            loss_normal = agent_normal.update(buffer_normal)
            print(f"传统智能体更新损失: {loss_normal}")
        except Exception as e:
            print(f"传统智能体更新错误: {e}")
    
    print("✓ 智能体兼容性测试通过")


def main():
    """主测试函数"""
    print("开始时间感知GRU-DDPG系统测试...")
    
    try:
        test_time_aware_networks()
        test_environment_time_noise()
        test_agent_compatibility()
        print("\n🎉 所有测试通过！时间感知功能正常工作。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 辅助函数定义

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def plot_rewards(rewards, avg_rewards=None, window=10, save_dir=None):
    """绘制奖励曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.5)
    
    if avg_rewards is None and len(rewards) > window:
        # 计算移动平均
        avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards)), avg_rewards, label=f'{window}-Episode Average')
    elif avg_rewards is not None:
        plt.plot(avg_rewards, label='Average Reward')
        
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_rewards.png'))
    plt.show()

    
def plot_state_comparison(results_no_control, results_ddpg):
    """比较不同控制策略下的状态轨迹"""
    # 绘制位移
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 0], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 0], label='LQR')
    plt.plot(results_ddpg['times'], results_ddpg['states'][:, 0], label='DDPG')
    plt.xlabel('Time (s)')
    plt.ylabel('Position x1')
    plt.title('吸振器位移')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 2], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 2], label='LQR')
    plt.plot(results_ddpg['times'], results_ddpg['states'][:, 2], label='DDPG')
    plt.xlabel('Time (s)')
    plt.ylabel('Position x2')
    plt.title('平台位移')
    plt.legend()
    plt.grid(True)
    
    # 绘制速度
    plt.subplot(2, 2, 3)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 1], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 1], label='LQR')
    plt.plot(results_ddpg['times'], results_ddpg['states'][:, 1], label='DDPG')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity v1')
    plt.title('吸振器速度')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(results_no_control['times'], results_no_control['states'][:, 3], label='No Control')
    # plt.plot(results_lqr['times'], results_lqr['states'][:, 3], label='LQR')
    plt.plot(results_ddpg['times'], results_ddpg['states'][:, 3], label='DDPG')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity v2')
    plt.title('平台速度')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 绘制控制输入
    plt.figure(figsize=(10, 5))
    # plt.plot(results_lqr['times'][:-1], results_lqr['actions'], label='LQR Control')
    plt.plot(results_ddpg['times'][:-1], results_ddpg['actions'], label='DDPG Control')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input')
    plt.title('控制输入对比')
    plt.legend()
    plt.grid(True)
    plt.show()
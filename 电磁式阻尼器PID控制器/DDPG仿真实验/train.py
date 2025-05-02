from tqdm import tqdm
import numpy as np
from typing import Callable, Dict, List, Any, Optional
from datetime import datetime
import os
from ddpg_agent import DDPGAgent, ReplayBuffer
from env import ElectromagneticDamperEnv

def train_ddpg(env: ElectromagneticDamperEnv, agent: DDPGAgent, replay_buffer: ReplayBuffer, 
              n_episodes=50, batch_size=64, min_buffer_size=1000, print_interval=10, save_interval=5, 
              save_path=None, r_func: Callable=None, 
              start_episode=0, initial_episode_rewards=None, load_previous=False, previous_model=None):
    """## 训练DDPG代理
    参数\n
    - env: 环境对象\n
    - agent: DDPG代理对象\n
    - replay_buffer: 经验回放池对象\n
    - n_episodes: 训练轮次，默认值为 50\n
    - batch_size: 批次大小，默认值为 64\n
    - min_buffer_size: 最小回放池大小，默认值为 1000\n
    - print_interval: 打印间隔，默认值为 10\n
    - save_path: 模型保存路径，默认值为 None\n
    - r_func: 奖励函数，默认值为 None\n
    - start_episode: 起始训练轮次，仅在继续训练时有效\n
    - initial_episode_rewards: 之前的奖励记录，仅在继续训练时有效\n
    - load_previous: 是否加载之前的模型，默认为 False\n
    - previous_model: 之前的模型路径，仅在 load_previous 为 True 时有效\n
    """
    # 记录训练情况
    episode_rewards = [] if initial_episode_rewards is None else initial_episode_rewards
    avg_rewards = []
    critic_losses = []
    actor_losses = []
    
    # 当前时间，用于模型命名
    current_time = datetime.now().strftime("%m%d_%H%M")
    
    # 训练循环
    for episode in tqdm(range(start_episode, start_episode + n_episodes), desc="训练轮次"):
        obs = env.reset() # 重置环境，获取初始观测值 (shape [1,])
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        num_updates = 0
        
        # 计算当前探索噪声的大小，使用线性衰减
        epsilon = max(1.0 - episode / ((start_episode + n_episodes) * 0.6), 0.1)
        
        done = False
        # tqdm_bar = tqdm(total=env.T/env.Ts, desc=f"Episode {episode+1}", leave=False)
        while not done:
            # tqdm_bar.update(1)
            # 选择动作 (基于当前观测值 obs)
            action = agent.select_action(obs, epsilon=epsilon)
            
            # 执行动作 (传入单个动作值)
            next_obs, done = env.step(action)

            # 计算奖励 (使用自定义奖励函数 r_func)
            if r_func is not None: reward = r_func(obs, action, next_obs)
            else: reward = 0.0  # 默认奖励为 0.0
            
            # 存储经验 (存储观测值)
            replay_buffer.add(obs, action, reward, next_obs, done) # 传递 done
            
            obs = next_obs # 更新观测值
            episode_reward += reward
            
            # 更新网络
            if len(replay_buffer) > min_buffer_size:
                critic_loss, actor_loss = agent.update(replay_buffer)
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                num_updates += 1
                
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        if num_updates > 0:
             critic_losses.append(episode_critic_loss / num_updates)
             actor_losses.append(episode_actor_loss / num_updates)
        else:
             critic_losses.append(0)
             actor_losses.append(0)
        
        # 打印训练进度
        if (episode + 1) % print_interval == 0:
            # 确保值是标量浮点数，然后格式化
            current_critic_loss = float(critic_losses[-1]) if critic_losses else 0.0
            current_actor_loss = float(actor_losses[-1]) if actor_losses else 0.0
            print(f"Episode: {episode+1}, Reward: {float(episode_reward):.2f}, Avg Reward: {float(avg_reward):.2f}, Avg Critic Loss: {current_critic_loss:.4f}, Avg Actor Loss: {current_actor_loss:.4f}, Epsilon: {float(epsilon):.2f}")
            
        # 保存模型和检查点
        if save_path and (episode + 1) % save_interval == 0:
            # 保存基本模型 (只有网络权重)
            model_name = f"{current_time}_ep{episode+1}"
            agent.save(f"{save_path}/ddpg_agent_{model_name}.pth")
            
            # 保存完整检查点 (包含所有训练状态)
            checkpoint_name = f"{current_time}_ep{episode+1}_checkpoint.pth"
            agent.save_checkpoint(
                f"{save_path}/{checkpoint_name}", 
                episode_rewards, 
                episode + 1
            )
            print(f"已保存模型和检查点: {model_name}")
    
    # 保存最终模型
    if save_path:
        final_model_name = f"{current_time}_ep{start_episode+n_episodes}_final"
        agent.save(f"{save_path}/ddpg_agent_{final_model_name}.pth")
        agent.save_checkpoint(
            f"{save_path}/{final_model_name}_checkpoint.pth", 
            episode_rewards, 
            start_episode + n_episodes
        )
        print(f"已保存最终模型和检查点: {final_model_name}")
        
    return {
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'final_model': f"{save_path}/ddpg_agent_{final_model_name}.pth" if save_path else None,
        'final_checkpoint': f"{save_path}/{final_model_name}_checkpoint.pth" if save_path else None
    }
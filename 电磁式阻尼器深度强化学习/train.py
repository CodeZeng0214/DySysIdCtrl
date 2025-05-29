from tqdm import tqdm
import numpy as np
from typing import Callable, Dict, List, Any, Optional
from datetime import datetime
import os
import logging
from ddpg_agent import DDPGAgent, ReplayBuffer
from env import ElectromagneticDamperEnv
from af import plot_test_data

def train_ddpg(env: ElectromagneticDamperEnv, agent: DDPGAgent, replay_buffer: ReplayBuffer, 
              n_episodes=200, min_buffer_size=1000, print_interval=5, save_interval=5, 
              save_path=None,rand_prob=0,
              start_episode=0, initial_episode_rewards=None):
    """## 训练DDPG代理
    参数\n
    - env: 环境对象\n
    - agent: DDPG代理对象\n
    - replay_buffer: 经验回放池对象\n
    - n_episodes: 训练轮次，默认值为 200\n
    - min_buffer_size: 最小回放池大小，默认值为 1000\n
    - print_interval: 打印间隔，默认值为 5\n
    - save_interval: 保存模型的间隔（轮数），默认值为 5\n
    - save_path: 训练项目保存路径，默认值为 None\n
    - r_func: 奖励函数，默认值为 None\n
    - start_episode: 起始训练轮次，仅在继续训练时有效\n
    - initial_episode_rewards: 之前的奖励记录，仅在继续训练时有效\n
    """
    # 记录训练情况
    episode_rewards = [] if initial_episode_rewards is None else initial_episode_rewards
    avg_rewards = []
    avg_critic_losses = []
    avg_actor_losses = []
    nc_data = env.run_simulation(controller=None) # 无控制的仿真数据
    
    # 当前时间，用于模型命名
    current_time = datetime.now().strftime("%m%d_%H%M")
    
    # 记录到日志
    logging.info(f"开始训练 - {datetime.now()}")
    logging.info(f"总轮次: {n_episodes}, 起始轮次: {start_episode}")
    
    # 创建保存路径
    save_plot_path = os.path.join(os.path.dirname(save_path), "plots")
    checkpoints_path = os.path.join(save_path) if save_path else None
    os.makedirs(checkpoints_path, exist_ok=True) if checkpoints_path else None
    logging.info(f"保存模型路径: {checkpoints_path}")
    
    # 创建奖励日志文件
    rewards_log_file = os.path.join(os.path.dirname(save_path), f"rewards_log{current_time}.csv") if save_path else None
    if rewards_log_file:
        with open(rewards_log_file, "w") as f:
            f.write("episode,reward,avg_reward,critic_loss,actor_loss,epsilon\n")
    
    # 训练循环
    for episode in tqdm(range(start_episode, n_episodes), desc="训练轮次"):
        env.reset() # 重置环境，获取初始观测值 (shape [1,])
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        num_updates = 0
        
        # 计算当前探索噪声的大小，使用线性衰减
        epsilon = max(1.0 - episode / ((start_episode + n_episodes) * 0.7), 0.1)
        
        done = False
        # tqdm_bar = tqdm(total=env.T/env.Ts, desc=f"Episode {episode+1}", leave=False)
        while not done:
            # tqdm_bar.update(1)
            # 选择动作 (基于当前观测值 obs)
            obs = env.get_observation() # 获取当前观测值 (shape [1,])
            action = agent.select_action(obs, epsilon=epsilon,rand_prob=rand_prob)
            
            # 执行动作 (传入单个动作值)
            next_obs, reward, done = env.step(action)
            
            # 存储经验 (存储观测值)
            replay_buffer.add(obs, action, reward, next_obs, done) # 传递 done
            
            episode_reward += reward
            
            # 更新网络
            if len(replay_buffer) > min_buffer_size:
                try:
                    critic_loss, actor_loss = agent.update(replay_buffer)
                except Exception as e:
                    print(f"更新网络时发生错误: {e}")
                    logging.error(f"更新网络时发生错误: {e}")
                    raise e
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                num_updates += 1
                
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        if num_updates > 0:
             avg_critic_losses.append(episode_critic_loss / num_updates)
             avg_actor_losses.append(episode_actor_loss / num_updates)
        else:
             avg_critic_losses.append(0)
             avg_actor_losses.append(0)
        
        # 获取当前训练指标
        current_critic_loss = float(avg_critic_losses[-1]) if avg_critic_losses else 0.0
        current_actor_loss = float(avg_actor_losses[-1]) if avg_actor_losses else 0.0
        
        # 记录到CSV
        if rewards_log_file:
            with open(rewards_log_file, "a") as f:
                f.write(f"{episode+1:>4d},{float(episode_reward):.6f},{float(avg_reward):.6f},{current_critic_loss:.6f},{current_actor_loss:.6f},{float(epsilon):.6f}\n")
                            
        # 保存模型训练的检查点
        if checkpoints_path and (episode + 1) % save_interval == 0:
            
            # 保存完整检查点 (包含所有训练状态)
            checkpoint_name = f"{current_time}_ep{episode+1}_checkpoint.pth"
            agent.model_name,_ = os.path.splitext(checkpoint_name)
            checkpoint_path = f"{checkpoints_path}/{checkpoint_name}"
            
            agent.save_checkpoint(
                checkpoint_path, 
                episode_rewards, 
                episode + 1
            )
            
        # 打印训练进度
        if (episode + 1) % print_interval == 0:
            # 确保值是标量浮点数，然后格式化
            log_msg = f"Episode: {episode+1:>4d}, Reward: {float(episode_reward):.2f}, Avg Reward: {float(avg_reward):.2f}, Avg Critic Loss: {current_critic_loss:.4f}, Avg Actor Loss: {current_actor_loss:.4f}, Epsilon: {float(epsilon):.2f}"
            print(log_msg)
            logging.info(log_msg)
            
            # 运行一次仿真并绘制结果
            test_data = env.run_simulation(controller=agent)
            plot_test_data(save_plot_path=save_plot_path,data=test_data,show=False, name=agent.model_name, nc_data=nc_data)
            save_msg = f"已保存模型数据: {checkpoint_name}"
            print(save_msg)
            # logging.info(save_msg)
        
    # 记录最终结果
    final_msg = f"训练完成，已保存最终模型: {checkpoint_name}, 训练轮次: {episode+1}, 最终奖励: {float(episode_reward):.2f}, 平均奖励: {float(avg_reward):.2f}"
    print(final_msg)
    logging.info(final_msg)
    logging.info(f"最终平均奖励: {float(avg_rewards[-1]):.4f}")
    
        
    return {
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards,
        'critic_losses': avg_critic_losses,
        'actor_losses': avg_actor_losses,
        'final_checkpoint': f"{checkpoints_path}/{checkpoint_name}" if checkpoints_path else None
    }
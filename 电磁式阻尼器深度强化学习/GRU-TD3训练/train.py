from tqdm import tqdm
import numpy as np
from typing import Callable, Dict, List, Any, Optional
from datetime import datetime
import os
import logging
from TD3 import TD3Agent, GruTD3Agent
from nn import ReplayBuffer, Gru_ReplayBuffer
from env import ElectromagneticDamperEnv
from af import plot_test_data

def train_td3(env: ElectromagneticDamperEnv, agent: TD3Agent, replay_buffer: ReplayBuffer, 
              n_episodes=200, min_buffer_size=1000, print_interval=5, save_interval=5, 
              save_path=None, rand_prob=0,
              start_episode=0, initial_episode_rewards=None):
    """## 训练TD3代理
    参数\n
    - env: 环境对象\n
    - agent: TD3代理对象\n
    - replay_buffer: 经验回放池对象\n
    - n_episodes: 训练轮次，默认值为 200\n
    - min_buffer_size: 最小回放池大小，默认值为 1000\n
    - print_interval: 打印间隔，默认值为 5\n
    - save_interval: 保存模型的间隔（轮数），默认值为 5\n
    - save_path: 训练项目保存路径，默认值为 None\n
    - rand_prob: 随机动作概率\n
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
    rewards_log_file = os.path.join(os.path.dirname(save_path), f"td3_rewards_log{current_time}.csv") if save_path else None
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
        while not done:
            # 选择动作 (基于当前观测值 obs)
            obs = env.get_observation() # 获取当前观测值 (shape [1,])
            action = agent.select_action(obs, epsilon=epsilon, rand_prob=rand_prob)
            
            # 执行动作
            next_obs, reward, done, _ = env.step(action)
            
            # 存储经验
            replay_buffer.add(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            
            # 更新网络
            if len(replay_buffer) > min_buffer_size:
                try:
                    critic_loss, actor_loss, total_critic_loss = agent.update(replay_buffer)
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
            
        # 日志和保存
        if rewards_log_file:
            with open(rewards_log_file, "a") as f:
                f.write(f"{episode},{episode_reward:.4f},{avg_reward:.4f},{avg_critic_losses[-1]:.6f},{avg_actor_losses[-1]:.6f},{epsilon:.4f}\n")
        
        # 打印进度
        if (episode + 1) % print_interval == 0:
            print(f"轮次 {episode+1}: 累计奖励 = {episode_reward:.2f}, 平均奖励 = {avg_reward:.2f}, "
                  f"Critic损失 = {avg_critic_losses[-1]:.6f}, Actor损失 = {avg_actor_losses[-1]:.6f}, "
                  f"探索率 = {epsilon:.4f}")
            logging.info(f"轮次 {episode+1}: 累计奖励 = {episode_reward:.2f}, 平均奖励 = {avg_reward:.2f}")
        
        # 保存检查点
        if (episode + 1) % save_interval == 0 and checkpoints_path:
            checkpoint_path = os.path.join(checkpoints_path, f"td3_{current_time}_ep{episode+1}_checkpoint.pth")
            agent.save_checkpoint(checkpoint_path, episode_rewards, episode + 1)
            print(f"保存检查点: {checkpoint_path}")
            
            # 保存并绘制当前模型的测试数据
            test_data = env.run_simulation(controller=agent)
            os.makedirs(save_plot_path, exist_ok=True)
            plot_test_data(save_plot_path, test_data, show=False, 
                          name=f"td3_{current_time}_ep{episode+1}_checkpoint", nc_data=nc_data)
    
    # 训练结束后保存最终模型
    if checkpoints_path:
        final_model_path = agent.save_model(checkpoints_path, n_episodes)
        print(f"训练完成！最终模型保存到: {final_model_path}")
        logging.info(f"训练完成！最终模型保存到: {final_model_path}")
    
    return episode_rewards, avg_rewards


def train_gru_td3(env: ElectromagneticDamperEnv, agent: GruTD3Agent, replay_buffer: Gru_ReplayBuffer, 
                  n_episodes=200, min_buffer_size=1000, print_interval=5, save_interval=5, 
                  save_path=None, rand_prob=0,
                  start_episode=0, initial_episode_rewards=None):
    """## 训练基于GRU的TD3代理
    参数\n
    - env: 环境对象\n
    - agent: GRU-TD3代理对象\n
    - replay_buffer: GRU经验回放池对象\n
    - n_episodes: 训练轮次，默认值为 200\n
    - min_buffer_size: 最小回放池大小，默认值为 1000\n
    - print_interval: 打印间隔，默认值为 5\n
    - save_interval: 保存模型的间隔（轮数），默认值为 5\n
    - save_path: 训练项目保存路径，默认值为 None\n
    - rand_prob: 随机动作概率\n
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
    logging.info(f"开始GRU-TD3训练 - {datetime.now()}")
    logging.info(f"总轮次: {n_episodes}, 起始轮次: {start_episode}")
    logging.info(f"序列长度: {agent.seq_len}, 使用时间输入: {agent.use_time_input}")
    
    # 创建保存路径
    save_plot_path = os.path.join(os.path.dirname(save_path), "plots")
    checkpoints_path = os.path.join(save_path) if save_path else None
    os.makedirs(checkpoints_path, exist_ok=True) if checkpoints_path else None
    logging.info(f"保存模型路径: {checkpoints_path}")
    
    # 创建奖励日志文件
    rewards_log_file = os.path.join(os.path.dirname(save_path), f"gru_td3_rewards_log{current_time}.csv") if save_path else None
    if rewards_log_file:
        with open(rewards_log_file, "w") as f:
            f.write("episode,reward,avg_reward,critic_loss,actor_loss,epsilon\n")
    
    # 训练循环
    for episode in tqdm(range(start_episode, n_episodes), desc="GRU-TD3训练轮次"):
        env.reset() # 重置环境
        agent.reset_history() # 重置代理状态历史
        replay_buffer.reset_history() # 重置回放池状态历史
        
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        num_updates = 0
        
        # 计算当前探索噪声的大小，使用线性衰减
        epsilon = max(1.0 - episode / ((start_episode + n_episodes) * 0.7), 0.1)
        
        done = False
        while not done:
            # 获取当前时间步长（可能包含噪声）
            current_dt = env.get_current_timestep()
            
            # 选择动作
            obs = env.get_observation()
            action = agent.select_action(obs, epsilon=epsilon, rand_prob=rand_prob, dt=current_dt)
            
            # 执行动作
            next_obs, reward, done, actual_dt = env.step(action, dt=current_dt)
            
            # 存储经验
            replay_buffer.add(obs, action, reward, next_obs, done, dt=actual_dt)
            
            episode_reward += reward
            
            # 更新网络
            if len(replay_buffer) > min_buffer_size:
                try:
                    critic_loss, actor_loss, total_critic_loss = agent.update(replay_buffer)
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
            
        # 日志和保存
        if rewards_log_file:
            with open(rewards_log_file, "a") as f:
                f.write(f"{episode},{episode_reward:.4f},{avg_reward:.4f},{avg_critic_losses[-1]:.6f},{avg_actor_losses[-1]:.6f},{epsilon:.4f}\n")
        
        # 打印进度
        if (episode + 1) % print_interval == 0:
            print(f"轮次 {episode+1}: 累计奖励 = {episode_reward:.2f}, 平均奖励 = {avg_reward:.2f}, "
                  f"Critic损失 = {avg_critic_losses[-1]:.6f}, Actor损失 = {avg_actor_losses[-1]:.6f}, "
                  f"探索率 = {epsilon:.4f}")
            logging.info(f"轮次 {episode+1}: 累计奖励 = {episode_reward:.2f}, 平均奖励 = {avg_reward:.2f}")
        
        # 保存检查点
        if (episode + 1) % save_interval == 0 and checkpoints_path:
            checkpoint_path = os.path.join(checkpoints_path, f"gru_td3_{current_time}_ep{episode+1}_checkpoint.pth")
            agent.save_checkpoint(checkpoint_path, episode_rewards, episode + 1)
            print(f"保存检查点: {checkpoint_path}")
            
            # 保存并绘制当前模型的测试数据
            test_data = env.run_simulation(controller=agent)
            os.makedirs(save_plot_path, exist_ok=True)
            plot_test_data(save_plot_path, test_data, show=False, 
                          name=f"gru_td3_{current_time}_ep{episode+1}_checkpoint", nc_data=nc_data)
    
    # 训练结束后保存最终模型
    if checkpoints_path:
        final_model_path = agent.save_model(checkpoints_path, n_episodes)
        print(f"训练完成！最终模型保存到: {final_model_path}")
        logging.info(f"训练完成！最终模型保存到: {final_model_path}")
    
    return episode_rewards, avg_rewards
from tqdm import tqdm
import numpy as np
from typing import Union
from datetime import datetime
import os
import logging
from TD3 import TD3Agent, Gru_TD3Agent
from nn import ReplayBuffer, Gru_ReplayBuffer
from env import ElectromagneticDamperEnv
from af import Datasets, plot_compare_no_control    

def train_td3(env: ElectromagneticDamperEnv, agent: Union[TD3Agent, Gru_TD3Agent], replay_buffer: Union[ReplayBuffer, Gru_ReplayBuffer], 
              n_episodes=200, min_buffer_size=1000, print_interval=5, save_interval=5, 
              project_path=None, save_checkpoint_path=None, save_plot_path=None, rand_prob=0,
              train_datasets:Datasets=None)->Datasets:
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
    - datasets: 数据集对象，默认值为 None\n
    """
    nc_datasets = env.run_simulation(controller=None, show_bar=False) # 无控制的仿真数据
    
    # 加载数据集
    train_datasets = Datasets() if train_datasets is None else train_datasets
    
    # 当前时间，用于模型命名
    project_time = datetime.now().strftime("%m%d_%H%M")
    
    # 记录到日志
    start_episode = train_datasets.current_episode
    logging.info(f"开始训练 - {datetime.now()}")
    logging.info(f"总轮次: {n_episodes}, 起始轮次: {start_episode}")

    # 创建奖励日志文件
    rewards_log_file = os.path.join(project_path, f"td3_rewards_log_{project_time}.csv") if project_path else None
    if rewards_log_file:
        with open(rewards_log_file, "w") as f:
            f.write(f"{'episode':>8}, {'rewards':>12}, {'critic_loss':>12}, {'actor_loss':>12}, {'epsilon':>12}\n")
    # 继承之前的数据
    if train_datasets.current_episode > 0:
        with open(rewards_log_file, "a") as f:
            for i in range(len(train_datasets.episode_rewards)):
                f.write(f"{i+1:>8}, {train_datasets.episode_rewards[i]:>12.4f}, {train_datasets.episode_critic_losses[i]:>12.6f}, {train_datasets.episode_actor_losses[i]:>12.6f}\n")

    # 训练循环
    for episode in tqdm(range(start_episode, n_episodes), desc="训练轮次"):
        env.reset() # 重置环境，获取初始观测值 (shape [1,])
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        num_updates = 0
        
        # 计算当前探索噪声的大小，使用线性衰减
        epsilon = max(1.0 - episode / ((start_episode + n_episodes) * 0.7), 0.1)
        # 计算当前探索噪声的大小，使用指数衰减
        epsilon = 0.1 + (1.0 - 0.1) * np.exp(-0.01 * episode)

        # 重置数据集的单回合历史记录
        train_datasets.current_episode = episode + 1
        train_datasets.reset_history() 
        agent.reset_history() # 重置代理的状态历史
        train_datasets.record_history(state=env.all_state.copy(), action=0.0, reward=0.0, dt=env.get_current_timestep(), time=env.time)
        if agent.delay_enabled: delay = max(1, int(np.random.normal(agent.delay_step, agent.delay_sigma)))
        else: delay = 1
        
        done = False
        while not done:
            # 选择动作 (基于当前观测值)
            state = env.get_observation() # 获取当前观测值 (shape [1,])
            
            # 如果使用时间步长感知或延迟感知，则将相关信息添加到状态中
            if agent.aware_dt: state = np.concatenate([state, np.array([train_datasets.dt_history[-1]])])
            if agent.aware_delay_time:
                if len(train_datasets.dt_history) < delay:
                    # 计算需要填充的时间步长数量
                    padding_dt_history = np.concatenate([[train_datasets.dt_history[0]] * max(0, delay - len(train_datasets.dt_history)), train_datasets.dt_history])
                else: padding_dt_history = train_datasets.dt_history
                delay_time = np.sum(padding_dt_history[-delay:])
                state = np.concatenate([state, np.array([delay_time])])
            
            env.state_history.append(state.copy())
            action = float(agent.select_action(env.state_history, add_noise=True, epsilon=epsilon, rand_prob=rand_prob, delay=delay))

            # 执行动作
            next_state, reward, done = env.step(action, dt=train_datasets.dt_history[-1])
            
            # 记录当前时间步的数据
            train_datasets.record_history(state=env.all_state.copy(), action=action, reward=reward, dt=env.get_current_timestep(), time=env.time)
            if agent.delay_enabled: delay = max(1, int(np.random.normal(agent.delay_step, agent.delay_sigma)))
            else: delay = 1
            if agent.aware_dt: next_state = np.concatenate([next_state, np.array([train_datasets.dt_history[-1]])])
            if agent.aware_delay_time:
                if len(train_datasets.dt_history) < delay:
                    # 计算需要填充的时间步长数量
                    padding_dt_history = np.concatenate([[train_datasets.dt_history[0]] * max(0, delay - len(train_datasets.dt_history)), train_datasets.dt_history])
                else: padding_dt_history = train_datasets.dt_history
                delay_time = np.sum(padding_dt_history[-delay:])
                next_state = np.concatenate([next_state, np.array([delay_time])])
                                   
            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新网络
            if len(replay_buffer) > min_buffer_size:
                try:
                    critic_loss, actor_loss, _ = agent.update(replay_buffer)
                except Exception as e:
                    print(f"更新网络时发生错误: {e}")
                    logging.error(f"更新网络时发生错误: {e}")
                    raise e
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                num_updates += 1
            
            episode_reward += reward
                
        # 记录本轮次的累计数据
        train_datasets.record_episode_data(episode_reward, 
                                           episode_actor_loss / num_updates if num_updates > 0 else 0, 
                                           episode_critic_loss / num_updates if num_updates > 0 else 0)

        # 保存csv数据文件
        if rewards_log_file:
            with open(rewards_log_file, "a") as f:
                f.write(f"{episode+1:>8}, {episode_reward:>12.4f}, {train_datasets.episode_critic_losses[-1]:>12.6f}, {train_datasets.episode_actor_losses[-1]:>12.6f}, {epsilon:>12.4f}\n")

        # 写入进度到日志
        if (episode + 1) % print_interval == 0:
            logging.info(f"轮次 {episode+1}: 累计奖励 = {episode_reward:.2f}")
        
        # 保存检查点
        if (episode + 1) % save_interval == 0 and save_checkpoint_path:
            train_datasets.checkpoint_name = f"{project_time}_ep{episode+1}_checkpoint"
            # print(agent.actor.net[-2].state_dict()['bias'])
            
            train_datasets.save_datasets(agent, save_checkpoint_path)

            # 保存当前模型的测试数据的控制图
            c_datasets = env.run_simulation(controller=agent, show_bar=False)
            c_datasets.checkpoint_name = f"{project_time}_ep{episode+1}_checkpoint"
            os.makedirs(save_plot_path, exist_ok=True)
            plot_compare_no_control(nc_datasets, c_datasets, save_path=save_plot_path, use_time_noise=env.use_dt_noise)

            # 保存当前轮次的探索过程数据图
            train_datasets.checkpoint_name = f"{project_time}_ep{episode+1}_datasets"
            plot_compare_no_control(nc_datasets, train_datasets, save_path=save_plot_path, use_time_noise=env.use_dt_noise)
            
    return train_datasets
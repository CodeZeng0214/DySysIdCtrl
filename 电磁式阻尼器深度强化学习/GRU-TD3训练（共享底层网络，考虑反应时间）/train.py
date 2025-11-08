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

def train_td3(env: ElectromagneticDamperEnv, agent: Union[TD3Agent, Gru_TD3Agent], 
              replay_buffer: Union[ReplayBuffer, Gru_ReplayBuffer], 
              n_episodes=200, min_buffer_size=1000, print_interval=5, save_interval=5, 
              project_path=None, save_checkpoint_path=None, save_plot_path=None, 
              rand_prob=0,
              train_datasets:Datasets=None,
              )->Datasets:
    """## 训练TD3代理（支持GRU预测器独立训练）
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
            f.write(f"{'episode':>8}, {'rewards':>12}, {'simu_reward':>12}, {'actor_loss':>12}, {'critic_loss':>12}, {'epsilon':>8}\n")
    # 继承之前的数据
    if train_datasets.current_episode > 0:
        with open(rewards_log_file, "a") as f:
            for i in range(len(train_datasets.episode_rewards)):
                f.write(f"{i+1:>8}, {train_datasets.episode_rewards[i]:>12.4f}, {train_datasets.episode_simu_rewards[i]:>12.4f}, {train_datasets.episode_actor_losses[i]:>12.4f}, {train_datasets.episode_critic_losses[i]:>12.4f}, {train_datasets.episode_epsilons[i]:>8.4f}\n")

    # 训练循环
    for episode in tqdm(range(start_episode, n_episodes), desc="训练轮次"):
        env.reset() # 重置环境，获取初始观测值 (shape [1,])
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        td3_num_updates = 0
        
        # 计算当前探索噪声的大小，使用线性衰减
        epsilon = max(1.0 - episode / ((start_episode + n_episodes) * 0.7), 0.1)
        # 计算当前探索噪声的大小，使用指数衰减
        epsilon = 0.1 + (1.0 - 0.1) * np.exp(-0.01 * episode)
        if episode >= n_episodes * 0.8: epsilon = 0 # 最后20%的轮次不使用探索噪声

        # 重置数据集的单回合历史记录
        train_datasets.current_episode = episode + 1
        train_datasets.reset_episode_data() 
        replay_buffer.reset_history()

        # 记录初始状态
        # 获取延迟时间步长
        if agent.delay_enabled: delay = max(1, int(np.random.normal(agent.delay_step, agent.delay_sigma))) 
        else: delay = 1
        train_datasets.record_history(state=env.all_state.copy(), action=0.0, reward=0.0, dt=env.get_current_timestep(), time=env.time, 
                                delay_time=env.Ts*delay)
        
        # 获取初始扩展状态
        state = extend_and_save_state(agent=agent, train_datasets=train_datasets, env=env, delay=delay)

        done = False
        step_count = 0
        while not done:
            # 选择动作
            action = float(agent.select_action(env.state_history, add_noise=(epsilon != 0), epsilon=epsilon, rand_prob=rand_prob, delay=delay))
            if episode + 1 <= 2: action = 0.0
            
            # 执行动作
            next_state, reward, done = env.step(action, dt=train_datasets.dt_history[-1])
            
            # 记录现在时间步的数据
            train_datasets.record_history(state=state[:6].copy(), action=action, reward=reward, dt=env.get_current_timestep(), time=env.time, 
                                        delay_time=state[-1] if agent.delay_enabled else 0.0)

            # 获取下一个扩展状态
            next_state = extend_and_save_state(agent=agent, train_datasets=train_datasets, env=env, delay=delay)

            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)
            
            # 将下一扩展状态更新为当前扩展状态
            state = next_state
            
            # 更新Actor和Critic网络
            if len(replay_buffer) > min_buffer_size:
                try:
                    critic_loss, actor_loss, _ = agent.update(replay_buffer)
                    td3_num_updates += 1
                except Exception as e:
                    print(f"更新网络时发生错误: {e}")
                    logging.error(f"更新网络时发生错误: {e}")
                    raise e
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                
            episode_reward += reward
            step_count += 1
        
        # 计算本轮次的平均损失
        episode_actor_loss = episode_actor_loss / td3_num_updates if td3_num_updates > 0 else 0
        episode_critic_loss = episode_critic_loss / td3_num_updates if td3_num_updates > 0 else 0

        # 运行当前策略的无动作探索仿真以评估性能
        c_datasets = env.run_simulation(controller=agent, show_bar=False)
        episode_simu_reward = c_datasets.reward_history.sum()
        
        # 记录本轮次的累计数据
        train_datasets.record_episode_data(episode_reward, episode_simu_reward, episode_actor_loss, episode_critic_loss, epsilon)

        # 保存检查点
        if ((episode + 1) % save_interval == 0 and save_checkpoint_path): #  or (episode_reward >= 0 and episode_simu_reward >= 0):
            train_datasets.checkpoint_name = f"{project_time}_ep{episode+1}_checkpoint"
            
            train_datasets.save_datasets(agent, save_checkpoint_path)

            # 保存当前模型的测试数据的控制图
            os.makedirs(save_plot_path, exist_ok=True)
            c_datasets.checkpoint_name = f"{project_time}_ep{episode+1}_simu_datasets"
            plot_compare_no_control(nc_datasets, c_datasets, plot_state=[3], save_path=save_plot_path, use_time_noise=env.use_dt_noise)

            # 保存当前轮次的探索过程数据图
            train_datasets.checkpoint_name = f"{project_time}_ep{episode+1}_expl_datasets"
            plot_compare_no_control(nc_datasets, train_datasets, plot_state=[3], save_path=save_plot_path, use_time_noise=env.use_dt_noise)

        # 保存csv数据文件
        if rewards_log_file:
            with open(rewards_log_file, "a") as f:
                f.write(f"{episode+1:>8}, {episode_reward:>12.4f}, {train_datasets.episode_simu_rewards[-1]:>12.4f}, {train_datasets.episode_actor_losses[-1]:>12.4f}, {train_datasets.episode_critic_losses[-1]:>12.4f}, {epsilon:>8.4f}\n")

        # 写入进度到日志
        if (episode + 1) % print_interval == 0:
            logging.info(f"轮次 {episode+1}: 累计奖励 = {episode_reward:.2f}, 仿真奖励 = {episode_simu_reward:.2f}, "
                         f"Actor损失 = {train_datasets.episode_actor_losses[-1]:.4f}, Critic损失 = {train_datasets.episode_critic_losses[-1]:.4f}, epsilon = {epsilon:.4f}")

    return train_datasets

def extend_and_save_state(agent: Gru_TD3Agent, train_datasets: Datasets, env: ElectromagneticDamperEnv, delay:int):
    """获取完整状态，可能包括延迟和时间步长信息\n
    保存状态到环境的状态历史中。"""
    state = env.get_observation() # 获取当前观测值 (shape [1,])
    
    # 如果使用时间步长感知或延迟感知，则将相关信息添加到状态中
    if agent.aware_dt: state = np.concatenate([state, np.array([train_datasets.dt_history[-1]])])
    if agent.aware_delay_time:
        if delay is None: raise ValueError("如果启用延迟感知，必须提供 delay 参数。")
        delay = max(1, int(np.random.normal(agent.delay_step, agent.delay_sigma))) if agent.delay_enabled else 1
        if len(train_datasets.dt_history) < delay:
            # 计算需要填充的时间步长数量
            padding_dt_history = np.concatenate([[train_datasets.dt_history[0]] * max(0, delay - len(train_datasets.dt_history)), train_datasets.dt_history])
        else: padding_dt_history = train_datasets.dt_history
        delay_time = np.sum(padding_dt_history[-delay:])
        state = np.concatenate([state, np.array([delay_time])])

    # 添加到状态历史
    env.state_history.append(state.copy())
    
    return state.copy()
import os
import time
from collections import deque
from typing import Optional, Callable
import numpy as np
from controller import BaseController
from buffer import ReplayBuffer
from data import EpisodeRecorder, TrainingHistory, save_checkpoint, load_checkpoint, latest_checkpoint, make_dirs, plot_data
from env import ElectromagneticDamperEnv
from tqdm import tqdm
import logging

def delayed_sequence(window: deque, delay_steps: int, seq_len: int) -> np.ndarray:
    """Delayed sequence for GRU policy input."""
    delay = max(0, min(delay_steps, len(window) - 1))
    hist = list(window)
    end = len(hist) - delay
    start = max(0, end - seq_len)
    view = hist[start:end]
    if len(view) < seq_len:
        pad = [hist[0]] * (seq_len - len(view))
        view = pad + view
    return np.stack(view, axis=0)


def delayed_obs(window: deque, delay_steps: int) -> np.ndarray:
    delay = max(0, min(delay_steps, len(window) - 1))
    return list(window)[-1 - delay]

def train(project_name: str, 
          env: ElectromagneticDamperEnv, controller: BaseController, buffer: ReplayBuffer,
          explore_noise_trend: str = 'linear',
          n_episodes=200, min_buffer_size=1000, save_interval=5,
          resume=False,
          state0: Optional[np.ndarray] = None,
          z_func: Optional[Callable] = None, f_func: Optional[Callable] = None,
          batch_size: int = 64
          ) -> None:
    """训练函数。\n
    参数：\n
        project_name: 项目名称，用于创建保存路径。
        env: 强化学习环境实例。
        controller: 强化学习控制器实例。
        buffer: 经验回放池实例。
        explore_noise_trend: 探索噪声变化趋势，'linear' 或 'exp'。
        n_episodes: 训练总回合数。
        min_buffer_size: 最小经验池大小，达到后开始训练。
        save_interval: 保存模型和日志的间隔回合数。
        resume: 是否从上次中断处恢复训练。
        state0: 初始状态，若为 None 则随机生成。
        z_func: 状态惩罚函数。
        f_func: 外部激励函数。
        batch_size: 训练批次大小。"""
    history = TrainingHistory()
    # 项目相关的路径设置
    project_path, ckpt_dir, plot_path = make_dirs(project_name)
    now_time = f"{time.strftime('%m%d_%H%M%S')}"

    nc_recorder = EpisodeRecorder()  # 用于记录无控制器时的表现
    nc_recorder = env.run_episode(controller=None, state0=state0, z_func=z_func, f_func=f_func)
    nc_x_values=nc_recorder.as_numpy(keys='time_history').reshape(-1, 1)[:,[0,0,0,0,0,0]]
    nc_y_values=nc_recorder.as_numpy(keys='state_history')[:,[0,1,2,3,4,5]]
    plot_data(x_values=nc_x_values, y_values=nc_y_values, sub_shape=(3,2),
              legends=[('吸振器位移',),('主结构位移',),('吸振器速度',),('主结构速度',),('吸振器加速度',),('主结构加速度',)], legend_loc='upper right', 
              sub_group=[(0,),(3,),(1,),(4,),(2,),(5,)],plot_title=f'{now_time}_初速度条件-环境无控制响应', save_path=plot_path, show=False)

    # 
    if resume:
        last = latest_checkpoint(ckpt_dir)
        if last:
            payload = load_checkpoint(last)
            controller.load_state(payload["agent"])
            history = TrainingHistory.from_dict(payload["history"])
            print(f"✅ Resumed from {last}")

    # 创建奖励日志文件
    rewards_log_file = os.path.join(project_path, f"td3_rewards_log_{time.strftime('%Y%m%d_%H%M%S')}.csv") if project_path else None
    if rewards_log_file:
        with open(rewards_log_file, "w") as f:
            f.write(f"{'episode':>8}, {'rewards':>12}, {'simu_reward':>12}, {'actor_loss':>12}, {'critic_loss':>12}, {'explore_noise':>8}\n")

    # 训练主循环
    for ep in tqdm(range(history.current_episode, n_episodes)):
        ep_recorder = EpisodeRecorder() # 记录当前回合数据

        # 回合相关变量初始化
        ep_reward_sum = 0.0
        ep_actor_loss_sum = 0.0
        ep_critic_loss_sum = 0.0
        updates = 0 # 控制器更新计数

        controller.reset() # 重置控制器状态
        env.reset(state0=state0, z_func=z_func, f_func=f_func) # 重置环境

        if explore_noise_trend == 'linear':
        # 计算当前探索噪声的大小，使用线性衰减
            explore_noise = max(1.0 - ep / ((history.current_episode + n_episodes) * 0.7), 0.1)
        elif explore_noise_trend == 'exp':
        # 计算当前探索噪声的大小，使用指数衰减
            explore_noise = 0.1 + (1.0 - 0.1) * np.exp(-0.01 * ep)
        else:
            explore_noise = 0.1 # 默认噪声值
        if ep >= n_episodes * 0.8: 
            explore_noise = 0 # 最后20%的轮次不使用探索噪声

        # 仿真轮次循环
        done = False
        while not done:
            obs = env.observe() # 获取观测值

            action = controller.select_action(obs=obs, noise_scale=explore_noise) # 选择动作
            next_obs, reward, done, info = env.step(action)

            buffer.add(obs, action, reward, next_obs, done, delay=info["delay_step"]) # 添加到经验回放池
            ep_reward_sum += reward

            ep_recorder.append(obs_history=obs.copy(), state_history=info["state"], action_history=action, reward_history=reward, 
                                time_history=info["time"], dt_history=info["dt"],  delay_time=info["delay_time"]) # 记录当前步数据
            
            # 控制器更新
            if len(buffer) > min_buffer_size:
                critic_loss, actor_loss = controller.update(replay_buffer=buffer, batch_size=batch_size)
                ep_critic_loss_sum += critic_loss
                ep_actor_loss_sum += actor_loss
                updates += 1 # 使控制器更新计数加一

        ep_actor_loss_avg = ep_actor_loss_sum / max(1, updates)
        ep_critic_loss_avg = ep_critic_loss_sum / max(1, updates)

        # 记录训练历史
        history.log(reward_history=ep_reward_sum, actor_loss_history=ep_actor_loss_avg, critic_loss_history=ep_critic_loss_avg, explore_noise_history=explore_noise)


        ep_sim_reward_sum = ep_reward_sum # 模拟运行环境（无噪声）的奖励总和
        if ep % save_interval == 0:
            # 运行有控制器的环境，记录数据
            c_recorder = env.run_episode(controller=controller, state0=state0, z_func=z_func, f_func=f_func)
            c_x_values = c_recorder.as_numpy(keys='time_history').reshape(-1, 1)
            c_y_values = c_recorder.as_numpy(keys='state_history')[:, [0, 1, 2, 3, 4, 5]]
            plot_data(x_values=c_x_values, y_values=np.concatenate((c_y_values,nc_y_values[:,[3]]), axis=1), 
                      legends=[('吸振器位移',),('无控制-主结构位移','GRUATD3控制-主结构位移'),('吸振器速度',),('主结构速度',),('吸振器加速度',),('主结构加速度',)], legend_loc='upper right',
                      sub_shape=(3, 2), sub_group=[(0,), (6,3), (1,), (4,), (2,), (5,)],
                      plot_title=f'{now_time}_初速度条件回合{ep}控制器响应', save_path=plot_path, show=False)
            c_action_values = c_recorder.as_numpy(keys='action_history').reshape(-1, 1)
            c_reward_values = c_recorder.as_numpy(keys='reward_history').reshape(-1, 1)
            c_delay_time_values = c_recorder.as_numpy(keys='delay_time').reshape(-1, 1)
            c_dt_values = c_recorder.as_numpy(keys='dt_history').reshape(-1, 1)
            plot_data(x_values=c_x_values, y_values=np.concatenate((c_action_values, c_reward_values, c_delay_time_values, c_dt_values), axis=1),
                      sub_shape=(2, 2), sub_group=[(0,), (1,), (2,), (3,)],
                      legends=[('动作',), ('奖励',), ('延迟时间',), ('时间步长',)], legend_loc='upper right',
                      plot_title=f'{now_time}_初速度条件回合{ep}控制器动作等', save_path=plot_path, show=False)
            ep_sim_reward_sum = c_recorder.as_numpy(keys='reward_history').sum() # 仿真奖励总和
            
            # 保存模型和训练当前的历史
            history.checkpoint_name = f"{time.strftime('%m%d_%H%M%S')}_ep{ep}ckpt"
            ckpt_path = os.path.join(ckpt_dir, f"{history.checkpoint_name}.pth")
            save_checkpoint(ckpt_path, controller.export_state(), ep_recorder, history)
            logging.info(f"💾 saved checkpoint {ckpt_path}")

        # 写入训练信息到csv日志文件
        if rewards_log_file:
            with open(rewards_log_file, "a") as f:
                
                f.write(f"{ep:>8}, {ep_reward_sum:>12.4f}, {ep_sim_reward_sum:>12.4f}, {ep_actor_loss_avg:>12.4f}, {ep_critic_loss_avg:>12.4f}, {explore_noise:>8.4f}\n")

    print("Training finished.")


if __name__ == "__main__":
    train()

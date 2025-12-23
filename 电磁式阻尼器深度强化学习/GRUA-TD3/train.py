import os
import time
from collections import deque
from typing import Optional, Callable
import numpy as np
from controller import BaseController
from buffer import ReplayBuffer
from data import EpisodeRecorder, TrainingHistory, save_checkpoint, load_checkpoint, latest_checkpoint, make_dirs, plot_data
from env import ElectromagneticDamperEnv

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
          explore_noise: float = 0.1,
          n_episodes=200, min_buffer_size=1000, save_interval=5,
          resume=False,
          state0: Optional[np.ndarray] = None,
          z_func: Optional[Callable] = None, f_func: Optional[Callable] = None,
          batch_size: int = 64
          ) -> None:
    
    history = TrainingHistory()
    # 项目相关的路径设置
    project_path, ckpt_dir, plt_dir = make_dirs(project_name)

    nc_recorder = EpisodeRecorder()  # 用于记录无控制器时的表现
    nc_recorder = env.run_episode(controller=None, state0=state0, z_func=z_func, f_func=f_func)

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
    for ep in range(history.current_episode, n_episodes):
        ep_recorder = EpisodeRecorder() # 记录当前回合数据

        # 回合相关变量初始化
        ep_reward_sum = 0.0
        ep_actor_loss_sum = 0.0
        ep_critic_loss_sum = 0.0
        updates = 0 # 控制器更新计数

        controller.reset() # 重置控制器状态
        env.reset(state0=state0, z_func=z_func, f_func=f_func) # 重置环境

        # 探索率更新
        if explore_noise is not None:
            explore_noise = max(0.1, explore_noise * 0.995)

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

        # 写入训练信息到csv日志文件
        if rewards_log_file:
            with open(rewards_log_file, "a") as f:
                f.write(f"{ep:>8}, {ep_reward_sum:>12.4f}, {ep_reward_sum:>12.4f}, {ep_actor_loss_avg:>12.4f}, {ep_critic_loss_avg:>12.4f}, {explore_noise:>8.4f}\n")

        if ep % save_interval == 0:
            history.checkpoint_name = f"ep{ep}" \
                f"_{time.strftime('%Y%m%d_%H%M%S')}"
            ckpt_path = os.path.join(ckpt_dir, f"{history.checkpoint_name}.pth")
            save_checkpoint(ckpt_path, controller.export_state(), ep_recorder, history)
            print(f"💾 saved checkpoint {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    train()

import os
import time
from collections import deque
from typing import Optional, Callable
import numpy as np
from controller import BaseController
from buffer import ReplayBuffer
from data import EpisodeRecorder, TrainingHistory, save_checkpoint, load_checkpoint, latest_checkpoint
from env import ElectromagneticDamperEnv
from fx import tolerance_reward, zero


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


def build_env():
    # Example physical parameters (same structure as旧代码)
    m = 1.0
    M = 15.0
    k_m = 30000.0
    k_M = 300000.0
    k_f = 100.0
    k_E = 0.0
    L = 0.0045
    R_m = 5.0
    c_m = 0.001
    c_M = 0.01

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-k_m / m, -c_m / m, k_m / m, c_m / m],
            [0.0, 0.0, 0.0, 1.0],
            [k_m / M, c_m / M, -(k_m + k_M) / M, -(c_m + c_M) / M],
        ]
    )
    B = np.array([[0.0], [k_f / m], [0.0], [-k_f / M]])
    C = np.array(
        [
            [-k_m / m, -c_m / m, k_m / m, c_m / m],
            [k_m / M, c_m / M, -(k_m + k_M) / M, -(c_m + c_M) / M],
        ]
    )
    D = np.array([[+k_f / m], [-k_f / M]])
    E = np.array([[0.0, 0.0, 0.0, c_M / M], [0.0, 0.0, 0.0, k_M / M]]).T
    F = np.array([[0.0], [0.0], [0.0], [1 / M]])

    env = ElectromagneticDamperEnv(
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
        F=F,
        Ts=0.001,
        T=1.0,
        state0=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        obs_indices=[0, 1, 2, 3, 4, 5],
        x1_limit=10000,
        use_dt_noise=False,
        dt_noise_std=0.1,
        delay_enabled=True,
        delay_mean_steps=3,
        delay_std_steps=1,
        include_dt_in_obs=False,
        include_delay_in_obs=True,
        z_func=zero,
        r_func=tolerance_reward(1e-3),
        f_func=zero,
    )
    return env


def train(env: ElectromagneticDamperEnv, controller: BaseController, buffer: ReplayBuffer,
          explore_noise: float = 0.1,
          n_episodes=200, min_buffer_size=1000, save_interval=5,
          project_path=None, resume=False,
          state0: Optional[np.ndarray] = None,
          z_func: Optional[Callable] = None, f_func: Optional[Callable] = None,
          batch_size: int = 64
          ) -> None:
    
    history = TrainingHistory()
    # 项目相关的路径设置
    os.makedirs(project_path, exist_ok=True)
    ckpt_dir = os.path.join(project_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    plt_dir = os.path.join(project_path, "plots")
    os.makedirs(plt_dir, exist_ok=True)

    # 
    if resume:
        last = latest_checkpoint(ckpt_dir)
        if last:
            payload = load_checkpoint(last)
            controller.load_state(payload["agent"])
            history = TrainingHistory.from_dict(payload["history"])
            print(f"✅ Resumed from {last}")

    # 探索率更新
    if explore_noise:
        explore_noise = max(0.1, explore_noise * 0.995)

    # 训练主循环
    for ep in range(history.current_episode, n_episodes):
        ep_recorder = EpisodeRecorder() # 记录当前回合数据

        controller.reset() # 重置控制器状态
        
        obs = env.reset()
        # buffer_len = seq_len + 50  # keep extra history to cover sampled delays
        # state_window = deque([obs.copy() for _ in range(buffer_len)], maxlen=buffer_len)
        # current_delay = env.delay_step

        ep_recorder = EpisodeRecorder()
        ep_reward_sum = 0.0
        ep_actor_loss_sum = 0.0
        ep_critic_loss_sum = 0.0
        updates = 0

        # 仿真轮次循环
        env.reset(state0=state0, z_func=z_func, f_func=f_func) # 重置环境
        done = False
        while not done:
            obs = env.observe() # 获取当前观测值

            # if arch == "gru":
            #     state_input = delayed_sequence(state_window, current_delay, seq_len)
            #     state_for_buffer = delayed_obs(state_window, current_delay)
            # else:
            #     state_input = delayed_obs(state_window, current_delay)
            #     state_for_buffer = state_input


            action = controller.select_action(obs=obs, epsilon=explore_noise) # 选择动作
            next_obs, reward, done, info = env.step(action)

            # state_window.append(next_obs)
            # next_delay = info.get("delay_step", 0)
            # next_state_view = delayed_obs(state_window, next_delay)

            # buffer.add(state_for_buffer, action, reward, next_state_view, done, delay=current_delay)

            ep_recorder.append(obs_history=obs.copy(), state_history=info["state"], action_history=action, reward_history=reward, 
                                time_history=info["time"], dt_history=info["dt"],  delay_time=info["delay_time"])
            # current_delay = next_delay

            ep_reward_sum += reward
            
            # 控制器更新
            if len(buffer) > min_buffer_size:
                critic_loss, actor_loss = controller.update(replay_buffer=buffer)
                ep_critic_loss_sum += critic_loss
                ep_actor_loss_sum += actor_loss
                updates += 1 # 使控制器更新计数加一

        ep_actor_loss_avg = ep_actor_loss_sum / max(1, updates)
        ep_critic_loss_avg = ep_critic_loss_sum / max(1, updates)

        history.log(reward_history=ep_reward_sum, actor_loss_history=ep_actor_loss_avg, critic_loss_history=ep_critic_loss_avg, explore_noise_history=explore_noise)
        
        if ep % save_interval == 0:
            history.checkpoint_name = f"ep{ep}" \
                f"_{time.strftime('%Y%m%d_%H%M%S')}"
            ckpt_path = os.path.join(ckpt_dir, f"{history.checkpoint_name}.pth")
            save_checkpoint(ckpt_path, controller.export_state(), ep_recorder, history)
            print(f"💾 saved checkpoint {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    train()

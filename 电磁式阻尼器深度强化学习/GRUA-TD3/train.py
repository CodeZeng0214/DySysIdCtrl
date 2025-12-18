import os
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

from agent import TD3Agent
from buffer import ReplayBuffer
from data import EpisodeRecorder, TrainingHistory, save_checkpoint, load_checkpoint, latest_checkpoint
from env import ElectromagneticDamperEnv
from fx import tolerance_reward, zero, sin_wave


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


def train(
    save_dir: str = "savedata/zz",
    n_episodes: int = 300,
    batch_size: int = 256,
    min_buffer: int = 5000,
    arch: str = "gru",
    seq_len: int = 36,
    hidden_dim: int = 128,
    gru_hidden: int = 64,
    policy_freq: int = 2,
    actor_lr: float = 2e-4,
    critic_lr: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.002,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    action_sigma: float = 0.15,
    clip_grad: float = 1.0,
    checkpoint_interval: int = 20,
    resume: bool = True,
) -> None:
    env = build_env()
    sample_obs = env.observe()
    state_dim = len(sample_obs)
    action_dim = 1

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=5.0,
        arch=arch,
        hidden_dim=hidden_dim,
        gru_hidden=gru_hidden,
        gru_layers=1,
        seq_len=seq_len,
        gamma=gamma,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_freq=policy_freq,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        clip_grad=clip_grad,
    )

    buffer = ReplayBuffer(state_dim=state_dim, capacity=int(5e5), seq_len=seq_len)
    history = TrainingHistory()

    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if resume:
        last = latest_checkpoint(ckpt_dir)
        if last:
            payload = load_checkpoint(last)
            agent.load_state(payload["agent"])
            history = TrainingHistory.from_dict(payload["history"])
            print(f"✅ Resumed from {last}")

    epsilon = 1.0
    for ep in range(history.current_episode, n_episodes):
        obs = env.reset()
        state_window = deque([obs.copy() for _ in range(seq_len)], maxlen=seq_len)
        ep_rec = EpisodeRecorder()
        ep_rec.append(state=env.all_state.copy(), action=0.0, reward=0.0, dt=env.last_dt, time=env.time, delay_time=env.delay_time)

        done = False
        step = 0
        ep_reward = 0.0
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        updates = 0

        while not done:
            state_input = np.stack(state_window, axis=0) if arch == "gru" else state_window[-1]
            action = agent.select_action(state_input, add_noise=True, noise_scale=epsilon)
            next_obs, reward, done, info = env.step(action)
            buffer.add(state_window[-1], action, reward, next_obs, done)

            ep_rec.append(state=env.all_state.copy(), action=action, reward=reward, dt=info["dt"], time=env.time, delay_time=info["delay_time"])
            ep_reward += reward
            state_window.append(next_obs)
            step += 1

            if len(buffer) > min_buffer:
                batch = buffer.sample(batch_size, use_sequence=(arch == "gru"))
                critic_loss, actor_loss = agent.update(batch, use_sequence=(arch == "gru"))
                critic_loss_sum += critic_loss
                actor_loss_sum += actor_loss
                updates += 1

        actor_loss_avg = actor_loss_sum / max(1, updates)
        critic_loss_avg = critic_loss_sum / max(1, updates)
        epsilon = max(0.1, epsilon * 0.995)

        history.log(reward=ep_reward, actor_loss=actor_loss_avg, critic_loss=critic_loss_avg, epsilon=epsilon)

        if (ep + 1) % checkpoint_interval == 0:
            history.checkpoint_name = f"ep{ep+1}" \
                f"_{time.strftime('%Y%m%d_%H%M%S')}"
            ckpt_path = os.path.join(ckpt_dir, f"{history.checkpoint_name}.pth")
            save_checkpoint(ckpt_path, agent.export_state(), ep_rec, history)
            print(f"💾 saved checkpoint {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    train()

import copy
from collections import deque
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import build_actor_critic, device
from fx import ACTION_BOUND


def _delayed_sequence(window: deque, delay_steps: int, seq_len: int) -> np.ndarray:
    delay = max(0, min(delay_steps, len(window) - 1))
    hist = list(window)
    end = len(hist) - delay
    start = max(0, end - seq_len)
    view = hist[start:end]
    if len(view) < seq_len:
        pad = [hist[0]] * (seq_len - len(view))
        view = pad + view
    return np.stack(view, axis=0)


def _delayed_obs(window: deque, delay_steps: int) -> np.ndarray:
    delay = max(0, min(delay_steps, len(window) - 1))
    return list(window)[-1 - delay]


class BaseController:
    """Interface for episode rollout controllers."""
    def __init__(self) -> None:
        self.obs_state_history = [] # 记录观测状态的历史

    def reset(self, first_obs: np.ndarray) -> None:
        self.obs_state_history = []
        raise NotImplementedError

    def select_action(self, obs: np.ndarray, obs_state_history=None) -> float:
        """基于当前观测值选择动作"""
        raise NotImplementedError

class PassiveController(BaseController):
    """Open-circuit passive damper (always zero control)."""

    def reset(self, first_obs: np.ndarray) -> None:
        pass

    def select_action(self, obs: np.ndarray) -> float:
        return 0.0


class PIDController(BaseController):
    """Simple PID on a chosen state index (defaults to x2 displacement at index 3)."""

    def __init__(self, kp: float, ki: float, kd: float, target: float = 0.0, state_index: int = 3, dt: float = 1e-3, u_max: float = ACTION_BOUND) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.state_index = state_index
        self.dt = dt
        self.u_max = u_max
        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self, first_obs: np.ndarray) -> None:
        self.integral = 0.0
        self.prev_err = float(self.target - first_obs[self.state_index])

    def select_action(self, obs: np.ndarray, delay_steps: int = 0) -> float:
        err = float(self.target - obs[self.state_index])
        self.integral += err * self.dt
        deriv = (err - self.prev_err) / self.dt if self.dt > 0 else 0.0
        self.prev_err = err
        u = self.kp * err + self.ki * self.integral + self.kd * deriv
        return float(np.clip(u, -self.u_max, self.u_max))


class TD3Agent:
    """TD3 with pluggable Actor/Critic (MLP or GRU)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float,
        arch: str = "mlp",
        hidden_dim: int = 128,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        seq_len: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        clip_grad: float = 0.0,
        use_attention: bool = True,
    ) -> None:
        self.arch = arch.lower()
        self.seq_len = max(1, seq_len)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_bound = action_bound
        self.clip_grad = clip_grad if clip_grad and clip_grad > 0 else None

        self.actor, self.critic1 = build_actor_critic(self.arch, state_dim, action_dim, hidden_dim, action_bound, gru_hidden, gru_layers, use_attention)
        _, self.critic2 = build_actor_critic(self.arch, state_dim, action_dim, hidden_dim, action_bound, gru_hidden, gru_layers, use_attention)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.total_it = 0

    # ------------------------------------------------------------------
    def select_action(self, state, add_noise: bool = True, noise_scale: float = 1.0) -> float:
        self.actor.eval()
        with torch.no_grad():
            if self.arch == "gru":
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = self.actor(state_tensor).cpu().numpy().flatten()
        self.actor.train()
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.policy_noise * noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return float(action[0])

    # ------------------------------------------------------------------
    def update(self, batch, use_sequence: bool, policy_update: bool = True) -> Tuple[float, float]:
        # delays are included in batch for logging/compatibility but not used by vanilla TD3
        states, actions, rewards, next_states, dones, _delays = batch
        self.total_it += 1

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.target_actor(next_states) + noise).clamp(-self.action_bound, self.action_bound)
            target_q1 = self.target_critic1(next_states, next_act) if self.arch == "mlp" else self.target_critic1(next_states, next_act)
            target_q2 = self.target_critic2(next_states, next_act) if self.arch == "mlp" else self.target_critic2(next_states, next_act)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + self.gamma * target_q * (1 - dones)

        # Critic update
        current_q1 = self.critic1(states, actions) if self.arch == "mlp" else self.critic1(states, actions)
        current_q2 = self.critic2(states, actions) if self.arch == "mlp" else self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip_grad)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip_grad)
        self.critic2_optim.step()

        actor_loss_value = 0.0
        if policy_update and self.total_it % self.policy_freq == 0:
            policy_actions = self.actor(states)
            actor_loss = -self.critic1(states, policy_actions).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
            self.actor_optim.step()
            actor_loss_value = float(actor_loss.item())

            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

        critic_loss_value = float((critic1_loss + critic2_loss).item() / 2)
        return critic_loss_value, actor_loss_value

    # ------------------------------------------------------------------
    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        for p_t, p_s in zip(tgt.parameters(), src.parameters()):
            p_t.data.copy_(self.tau * p_s.data + (1 - self.tau) * p_t.data)

    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, object]:
        return {
            "arch": self.arch,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic1_optim": self.critic1_optim.state_dict(),
            "critic2_optim": self.critic2_optim.state_dict(),
            "total_it": self.total_it,
        }

    def load_state(self, payload: Dict[str, object]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.target_actor.load_state_dict(payload["target_actor"])
        self.target_critic1.load_state_dict(payload["target_critic1"])
        self.target_critic2.load_state_dict(payload["target_critic2"])
        self.actor_optim.load_state_dict(payload["actor_optim"])
        self.critic1_optim.load_state_dict(payload["critic1_optim"])
        self.critic2_optim.load_state_dict(payload["critic2_optim"])
        self.total_it = int(payload.get("total_it", 0))


class TD3Controller(BaseController):
    """Wrapper around TD3Agent that handles delayed inputs for MLP or GRU."""

    def __init__(self, agent: TD3Agent) -> None:
        self.agent = agent
        buffer_len = agent.seq_len + 50
        self.window: deque = deque(maxlen=buffer_len)

    def reset(self, first_obs: np.ndarray) -> None:
        self.window.clear()
        for _ in range(self.window.maxlen):
            self.window.append(first_obs.copy())

    def select_action(self, obs: np.ndarray, delay_steps: int = 0) -> float:
        self.window.append(obs.copy())
        if self.agent.arch == "gru":
            state_input = _delayed_sequence(self.window, delay_steps, self.agent.seq_len)
        else:
            state_input = _delayed_obs(self.window, delay_steps)
        return self.agent.select_action(state_input, add_noise=True)

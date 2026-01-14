from typing import Dict, Tuple
from buffer import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import build_actor_critic, build_ppo_actor_critic, device
from fx import ACTION_BOUND



class BaseController:
    """Interface for episode rollout controllers."""
    def __init__(self) -> None:
        self.obs_state_history = []  # 记录观测状态的历史
        self.arch = "base" # 控制器架构类型

    def reset(self, first_obs: np.ndarray) -> None:
        self.obs_state_history = []
        raise NotImplementedError

    def select_action(self, obs: np.ndarray, noise_scale: float = 0.0) -> float:
        """基于当前观测值选择动作，可噪声尺度"""
        raise NotImplementedError
    
    def update(self, replay_buffer: ReplayBuffer, batch_size: int):
        """基于重放缓冲区数据更新控制器参数"""
        raise NotImplementedError
        
    def export_state(self) -> Dict[str, object]:
        """导出控制器状态以进行检查点保存"""
        raise NotImplementedError

    
    def load_state(self, state_dict: Dict[str, object]) -> None:
        """从检查点加载控制器状态"""
        raise NotImplementedError

    

class PassiveController(BaseController):
    """Open-circuit passive damper (always zero control)."""

    def reset(self, first_obs: np.ndarray) -> None:
        pass

    def select_action(self, obs: np.ndarray, delay_steps: int = 0, noise_scale: float = 1.0) -> float:
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

    def select_action(self, obs: np.ndarray, delay_steps: int = 0, noise_scale: float = 1.0) -> float:
        err = float(self.target - obs[self.state_index])
        self.integral += err * self.dt
        deriv = (err - self.prev_err) / self.dt if self.dt > 0 else 0.0
        self.prev_err = err
        u = self.kp * err + self.ki * self.integral + self.kd * deriv
        return float(np.clip(u, -self.u_max, self.u_max))


class TD3Controller(BaseController):
    """TD3 控制器，内置延迟历史对齐，可用于 MLP/GRU/Attention 结构。"""

    def __init__(
        self, arch: str = "mlp", 
        norm: bool = False, simple_nn: bool = False,
        state_dim: int = 6, action_dim: int = 1, action_bound: float = 5.0,
        gru_hidden: int = 64, gru_layers: int = 1, hidden_dim: int = 128, 
        seq_len: int = 1, fc_seq_len: int = 5,
        actor_lr: float = 2e-06, critic_lr: float = 1e-05, clip_grad: float = 1.0, tau: float = 0.002,
        gamma: float = 0.99, policy_noise: float = 0.2, noise_clip: float = 0.5, policy_freq: int = 2, 
        ) -> None:
        super().__init__()
        self.action_bound = action_bound # 动作边界
        self.arch = arch.lower() # 控制器架构类型
        self.seq_len = max(1, seq_len) # 序列长度，至少为1
        self.gamma = gamma # 折扣因子
        self.tau = tau # 软更新系数
        self.policy_noise = policy_noise # 策略噪声标准差
        self.noise_clip = noise_clip # 策略噪声裁剪范围
        self.policy_freq = policy_freq # 策略更新频率
        self.clip_grad = clip_grad if clip_grad and clip_grad > 0 else None # 梯度裁剪阈值

        self.actor, self.critic1 = build_actor_critic(self.arch, state_dim, action_dim, hidden_dim, action_bound, gru_hidden, gru_layers, fc_seq_len=fc_seq_len, norm=norm, simple_nn=simple_nn)
        _, self.critic2 = build_actor_critic(self.arch, state_dim, action_dim, hidden_dim, action_bound, gru_hidden, gru_layers, fc_seq_len=fc_seq_len, norm=norm, simple_nn=simple_nn)

        # 使用深拷贝创建目标网络，隔离网络的权重
        self.target_actor,self.target_critic1 = build_actor_critic(self.arch, state_dim, action_dim, hidden_dim, action_bound, gru_hidden, gru_layers, fc_seq_len=fc_seq_len, norm=norm, simple_nn=simple_nn)
        _, self.target_critic2 = build_actor_critic(self.arch, state_dim, action_dim, hidden_dim, action_bound, gru_hidden, gru_layers, fc_seq_len=fc_seq_len, norm=norm, simple_nn=simple_nn)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.total_it = 0 # 总更新迭代计数

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.obs_state_history = []  # 重置观测状态历史

    # ------------------------------------------------------------------
    def _policy_action(self, state: np.ndarray, add_noise: bool = True, noise_scale: float = 1.0) -> float:
        """基于当前状态序列计算动作，支持添加噪声"""
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = self.actor.forward(state_tensor).cpu().numpy().flatten()
        self.actor.train()
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.policy_noise * noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return float(action[0])

    def select_action(self, obs: np.ndarray, noise_scale: float = 0.0) -> float:
        """基于当前观测值选择动作，自动处理延迟历史对齐和噪声添加"""
        self.obs_state_history.append(obs.copy()) # 记录观测状态历史
        if self.arch == "mlp":
            state_input = obs
        elif self.arch == "seq":
            state_input = self.obs_state_history[-self.seq_len:] # 取最近 seq_len 个观测状态
        state_input = np.array(state_input, dtype=np.float32)
        return self._policy_action(state_input, add_noise=True, noise_scale=noise_scale)

    # ------------------------------------------------------------------
    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> Tuple[float, float]:
        """更新Actor和Critic网络"""
        self.total_it += 1

        use_seq = (False or self.arch == "seq")
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size=batch_size, use_sequence=use_seq)

        with torch.no_grad():
            # 目标策略平滑正则化
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor.forward(next_states) + noise).clamp(-self.action_bound, self.action_bound)
            
            # 计算目标Q值，取两个Critic的最小值
            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + self.gamma * target_q * (1 - dones)
        
        # 2. 更新两个Critic网络
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        # 梯度裁剪
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip_grad)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        # 梯度裁剪
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip_grad)
        self.critic2_optim.step()

        # 3. 延迟策略更新
        actor_loss_value = 0.0
        if self.total_it % self.policy_freq == 0:
            # 更新Actor网络
            policy_actions = self.actor(states)
            actor_loss: torch.Tensor = -self.critic1(states, policy_actions).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            # 梯度裁剪
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
            self.actor_optim.step()
            # 软更新目标网络
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
            
            actor_loss_value = float(actor_loss.item())
        critic_loss_value = float((critic1_loss + critic2_loss).item() / 2)
        return critic_loss_value, actor_loss_value

    # ------------------------------------------------------------------
    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        """软更新目标网络参数"""
        for p_t, p_s in zip(tgt.parameters(), src.parameters()):
            p_t.data.copy_(self.tau * p_s.data + (1 - self.tau) * p_t.data)

    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, object]:
        """导出控制器状态字典，用于保存模型。"""
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
        """从状态字典加载控制器模型参数。"""
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


class PPOController(BaseController):
    """PPO 控制器，支持 MLP/GRU/Attention 结构。
    
    与 TD3Controller 接口兼容，可直接用于现有的 train.py 训练循环。
    采用与 TD3 相同的更新逻辑：每个时间步从 ReplayBuffer 采样并更新。
    """

    def __init__(
        self, arch: str = "mlp",
        norm: bool = False, simple_nn: bool = False,
        state_dim: int = 6, action_dim: int = 1, action_bound: float = 5.0,
        gru_hidden: int = 64, gru_layers: int = 1, hidden_dim: int = 128,
        seq_len: int = 32, fc_seq_len: int = 4,
        actor_lr: float = 3e-4, critic_lr: float = 1e-3, 
        gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2, entropy_coef: float = 0.01, value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5, n_epochs: int = 4, mini_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.action_bound = action_bound
        self.arch = arch.lower()
        self.seq_len = max(1, seq_len)
        self.gamma = gamma  # 折扣因子
        self.gae_lambda = gae_lambda  # GAE lambda 参数
        self.clip_epsilon = clip_epsilon  # PPO 裁剪系数
        self.entropy_coef = entropy_coef  # 熵正则化系数
        self.value_loss_coef = value_loss_coef  # 价值损失系数
        self.max_grad_norm = max_grad_norm  # 梯度裁剪阈值
        self.n_epochs = n_epochs  # 每次更新的 epoch 数（K_epochs）
        self.mini_batch_size = mini_batch_size  # mini-batch 大小
        # 构建 Actor 和 Critic 网络
        self.actor, self.critic = build_ppo_actor_critic(
            self.arch, state_dim, action_dim, hidden_dim, action_bound,
            gru_hidden, gru_layers, fc_seq_len=fc_seq_len, norm=norm, simple_nn=simple_nn
        )

        # 使用统一的优化器
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])
        
        self.total_it = 0  # 总更新迭代计数

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """重置控制器状态（每个 episode 开始时调用）"""
        self.obs_state_history = []

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, noise_scale: float = 0.0) -> float:
        """基于当前观测值选择动作，接口与 TD3 兼容
        
        Args:
            obs: 当前观测值
            noise_scale: 探索噪声比例（0.0 表示确定性动作）
            
        Returns:
            action: 选择的动作值（float）
        """
        self.obs_state_history.append(obs.copy())
        
        if self.arch == "mlp":
            state_input = obs
        elif self.arch == "seq":
            state_input = self.obs_state_history[-self.seq_len:]
        state_input = np.array(state_input, dtype=np.float32)
        
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state_input, dtype=torch.float32, device=device).unsqueeze(0)
            
            # 根据 noise_scale 决定是否使用确定性动作
            deterministic = (noise_scale == 0.0)
            action, _ = self.actor.get_action_and_log_prob(state_tensor, deterministic=deterministic)
            
        self.actor.train()
        
        action_np = float(action.cpu().numpy().flatten()[0])
        return action_np

    # ------------------------------------------------------------------
    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> Tuple[float, float]:
        """更新 Actor 和 Critic 网络，接口与 TD3 兼容
        
        采用与 TD3 相同的更新逻辑：每个时间步从 ReplayBuffer 采样并更新。
        
        Args:
            replay_buffer: 经验回放池
            batch_size: 批次大小（每次更新采样的样本数）
            
        Returns:
            (critic_loss, actor_loss): 损失值元组
        """
        self.total_it += 1

        # 从经验池采样，与 TD3 相同
        use_seq = (False or self.arch == "seq")
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size=batch_size, use_sequence=use_seq
        )

        # 计算 GAE 优势估计和目标回报
        with torch.no_grad():
            # 使用当前 critic 计算价值
            values = self.critic(states).squeeze(-1)  # [batch_size]
            next_values = self.critic(next_states).squeeze(-1)  # [batch_size]
            
            # 计算 TD 误差 (deltas)
            rewards_flat = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
            dones_flat = dones.squeeze(-1) if dones.dim() > 1 else dones
            deltas = rewards_flat + self.gamma * next_values * (1 - dones_flat) - values
            
            # 计算 GAE（简化版，假设 batch 内样本独立）
            advantages = deltas  # 简化为单步 TD 误差
            returns = rewards_flat + self.gamma * next_values * (1 - dones_flat)
            
            # 标准化优势（增强数值稳定性）
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std < 1e-6:  # 如果标准差太小，不标准化
                advantages = advantages - adv_mean
            else:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # 准备数据张量
        advantages_tensor = advantages.unsqueeze(1)  # [batch_size, 1]
        returns_tensor = returns.unsqueeze(1)  # [batch_size, 1]
        
        # 多次 epoch 更新（K_epochs）
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            # 每个epoch开始时重新计算old_log_probs，防止梯度爆炸
            with torch.no_grad():
                old_log_probs, _ = self.actor.evaluate_actions(states, actions)
            
            # 随机打乱索引
            indices = torch.randperm(batch_size, device=device)
            
            # Mini-batch 更新
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                idx = indices[start:end]
                
                # 获取当前批次的数据
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages_tensor[idx]
                batch_returns = returns_tensor[idx]
                
                # 计算新策略的 log_prob 和熵
                new_log_probs, entropy = self.actor.evaluate_actions(batch_states, batch_actions)
                
                # 计算重要性采样比率（添加数值稳定性检查）
                log_ratio = new_log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, -20, 20)  # 防止exp溢出
                ratios = torch.exp(log_ratio)
                
                # 计算剪切目标函数（PPO-Clip）
                surr1 = ratios * batch_advantages  # 未剪切的目标
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()  # 策略损失
                
                # 计算价值函数损失
                batch_values = self.critic(batch_states)
                value_loss = F.mse_loss(batch_values, batch_returns)
                
                # 总损失 = 策略损失 + 价值损失系数 * 价值损失 - 熵系数 * 熵
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 执行梯度下降
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                total_actor_loss += policy_loss.item()
                total_critic_loss += value_loss.item()
                n_updates += 1
        
        avg_actor_loss = total_actor_loss / max(1, n_updates)
        avg_critic_loss = total_critic_loss / max(1, n_updates)
        
        return avg_critic_loss, avg_actor_loss

    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, object]:
        """导出控制器状态字典，用于保存模型。"""
        return {
            "arch": self.arch,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state(self, payload: Dict[str, object]) -> None:
        """从状态字典加载控制器模型参数。"""
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.total_it = int(payload.get("total_it", 0))


def build_controller(PARAMS, type:str):
    if type == "TD3":
        controller = TD3Controller(**PARAMS)
    elif type == "PPO":
        controller = PPOController(**PARAMS)
    else:
        raise ValueError
    return controller
    

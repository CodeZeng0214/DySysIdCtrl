## TD3 算法定义的函数

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from nn import Actor, Critic, ReplayBuffer, Gru_Actor, Gru_Critic, Gru_ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## TD3 代理
class TD3Agent:
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, sigma=0.2, clip_grad=False):
        """TD3参数\n
        state_dim: 状态维度\n
        action_dim: 动作维度\n
        hidden_dim: 隐藏层维度\n
        action_bound: 动作范围\n
        actor_lr: Actor网络学习率\n
        critic_lr: Critic网络学习率\n
        gamma: 折扣因子\n
        tau: 软更新参数\n
        policy_noise: 目标策略平滑正则化噪声\n
        noise_clip: 噪声裁剪范围\n
        policy_freq: 策略更新频率\n
        sigma: 探索噪声标准差\n
        clip_grad: 是否使用梯度裁剪\n
        """
        # 初始化参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau      # 软更新参数
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.sigma = sigma  # 探索噪声
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.clip_grad = clip_grad
        self.model_name = None
        
        self.total_it = 0  # 总迭代次数
        
        # 网络初始化
        self.actor = Actor(state_dim, action_dim, hidden_dim, action_bound).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        self.target_actor = Actor(state_dim, action_dim, hidden_dim, action_bound).to(device)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器设置
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
    def select_action(self, state: np.ndarray, add_noise=True, epsilon=1.0, rand_prob=0.05) -> np.ndarray:
        """选择动作，支持探索"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_tensor: torch.Tensor = self.actor(state)
            action_np: np.ndarray = action_tensor.cpu().detach().numpy()
            action = action_np.flatten()
            
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.sigma * epsilon, size=self.action_dim)
            action += noise
            if np.random.random() < rand_prob:
                action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        return np.clip(action, -self.action_bound, self.action_bound)
    
    def update(self, replay_buffer: ReplayBuffer) -> Tuple[float, float, float]:
        """更新Actor和Critic网络"""
        if len(replay_buffer) < replay_buffer.batch_size:
            return 0.0, 0.0, 0.0
        
        self.total_it += 1
        
        # 1. 从回放池中采样
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        
        with torch.no_grad():
            # 目标策略平滑正则化
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.action_bound, self.action_bound)
            
            # 计算目标Q值，取两个Critic的最小值
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + self.gamma * target_q * (1 - dones)
            
        # 2. 更新两个Critic网络
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=10)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10)
        self.critic2_optimizer.step()
        
        critic_loss = (critic1_loss + critic2_loss) / 2
        actor_loss = 0.0
        
        # 3. 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            # 更新Actor网络
            policy_actions = self.actor(states)
            actor_loss = -self.critic1(states, policy_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss, (critic1_loss.item() + critic2_loss.item()) / 2
    
    def _soft_update(self, source, target):
        """软更新目标网络参数"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
            
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
        }, path)
        
    def save_checkpoint(self, path, episode_rewards, current_episode):
        """保存包含训练状态的完整检查点"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'current_episode': current_episode,
            'total_it': self.total_it,
        }, path)
        
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        
    def load_checkpoint(self, path):
        """加载检查点并返回训练状态"""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        
        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if 'total_it' in checkpoint:
            self.total_it = checkpoint['total_it']
        
        episode_rewards = checkpoint.get('episode_rewards', [])
        current_episode = checkpoint.get('current_episode', 0)
        
        return episode_rewards, current_episode


## 基于GRU的TD3代理
class GruTD3Agent:
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, sigma=0.2, 
                 clip_grad=False, seq_len=10, num_layers=2, use_time_input=False):
        """基于GRU的TD3代理参数\n
        state_dim: 状态维度\n
        action_dim: 动作维度\n
        hidden_dim: 隐藏层维度\n
        action_bound: 动作范围\n
        actor_lr: Actor网络学习率\n
        critic_lr: Critic网络学习率\n
        gamma: 折扣因子\n
        tau: 软更新参数\n
        policy_noise: 目标策略平滑正则化噪声\n
        noise_clip: 噪声裁剪范围\n
        policy_freq: 策略更新频率\n
        sigma: 探索噪声标准差\n
        clip_grad: 是否使用梯度裁剪\n
        seq_len: 序列长度\n
        num_layers: GRU层数\n
        use_time_input: 是否使用时间输入\n
        """
        # 初始化参数
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.sigma = sigma
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.clip_grad = clip_grad
        self.seq_len = seq_len
        self.use_time_input = use_time_input
        self.model_name = None
        
        self.total_it = 0
        
        # GRU网络初始化
        self.actor = Gru_Actor(state_dim, action_dim, hidden_dim, action_bound, seq_len, num_layers, use_time_input).to(device)
        self.critic1 = Gru_Critic(state_dim, action_dim, hidden_dim, seq_len, num_layers, use_time_input).to(device)
        self.critic2 = Gru_Critic(state_dim, action_dim, hidden_dim, seq_len, num_layers, use_time_input).to(device)
        
        self.target_actor = Gru_Actor(state_dim, action_dim, hidden_dim, action_bound, seq_len, num_layers, use_time_input).to(device)
        self.target_critic1 = Gru_Critic(state_dim, action_dim, hidden_dim, seq_len, num_layers, use_time_input).to(device)
        self.target_critic2 = Gru_Critic(state_dim, action_dim, hidden_dim, seq_len, num_layers, use_time_input).to(device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器设置
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # 用于维护状态历史的缓冲区
        self.state_history = []
        self.time_history = []
        
    def reset_history(self):
        """重置状态历史，在新的episode开始时调用"""
        self.state_history = []
        self.time_history = []
        
    def select_action(self, state: np.ndarray, add_noise=True, epsilon=1.0, rand_prob=0.05, dt=None) -> np.ndarray:
        """选择动作，支持探索"""
        current_dt = dt
        
        # 更新状态历史
        self.state_history.append(state.copy())
        if self.use_time_input:
            self.time_history.append(current_dt)
        
        # 如果历史长度不够，使用零填充或重复当前状态
        if len(self.state_history) < self.seq_len:
            padded_history = [state] * (self.seq_len - len(self.state_history)) + self.state_history
            if self.use_time_input:
                padded_time_history = [current_dt] * (self.seq_len - len(self.time_history)) + self.time_history
        else:
            padded_history = self.state_history[-self.seq_len:]
            if self.use_time_input:
                padded_time_history = self.time_history[-self.seq_len:]

        # 构建状态序列
        state_seq = np.array(padded_history)
        state_seq_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        if self.use_time_input:
            time_seq = np.array(padded_time_history)
            time_seq_tensor = torch.tensor(time_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if self.use_time_input:
                action_tensor: torch.Tensor = self.actor(state_seq_tensor, time_seq_tensor)
            else:
                action_tensor: torch.Tensor = self.actor(state_seq_tensor)
            action_np = action_tensor.cpu().detach().numpy()
            action = action_np.flatten()
            
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.sigma * epsilon, size=self.action_dim)
            action += noise
            if np.random.random() < rand_prob:
                action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        return np.clip(action, -self.action_bound, self.action_bound)
    
    def update(self, replay_buffer: Gru_ReplayBuffer) -> Tuple[float, float, float]:
        """更新Actor和Critic网络"""
        if len(replay_buffer) < replay_buffer.batch_size:
            return 0.0, 0.0, 0.0
        
        self.total_it += 1
        
        # 1. 从回放池中采样
        state_seqs, actions, rewards, next_state_seqs, dones, time_seqs, next_time_seqs = replay_buffer.sample()
        
        with torch.no_grad():
            # 目标策略平滑正则化
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            if self.use_time_input:
                next_actions = (self.target_actor(next_state_seqs, next_time_seqs) + noise).clamp(-self.action_bound, self.action_bound)
                target_q1 = self.target_critic1(next_state_seqs, next_actions, next_time_seqs)
                target_q2 = self.target_critic2(next_state_seqs, next_actions, next_time_seqs)
            else:
                next_actions = (self.target_actor(next_state_seqs) + noise).clamp(-self.action_bound, self.action_bound)
                target_q1 = self.target_critic1(next_state_seqs, next_actions)
                target_q2 = self.target_critic2(next_state_seqs, next_actions)
            
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + self.gamma * target_q * (1 - dones)
            
        # 2. 更新两个Critic网络
        if self.use_time_input:
            current_q1 = self.critic1(state_seqs, actions, time_seqs)
            current_q2 = self.critic2(state_seqs, actions, time_seqs)
        else:
            current_q1 = self.critic1(state_seqs, actions)
            current_q2 = self.critic2(state_seqs, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=10)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10)
        self.critic2_optimizer.step()
        
        critic_loss = (critic1_loss + critic2_loss) / 2
        actor_loss = 0.0
        
        # 3. 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            if self.use_time_input:
                policy_actions = self.actor(state_seqs, time_seqs)
                actor_loss = -self.critic1(state_seqs, policy_actions, time_seqs).mean()
            else:
                policy_actions = self.actor(state_seqs)
                actor_loss = -self.critic1(state_seqs, policy_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss, (critic1_loss.item() + critic2_loss.item()) / 2
    
    def _soft_update(self, source, target):
        """软更新目标网络参数"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
            
    def save_model(self, save_path, episode):
        """保存模型"""
        import os
        from datetime import datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gru_td3_model_ep{episode}_{current_time}.pth"
        full_path = os.path.join(save_path, filename)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'episode': episode,
            'total_it': self.total_it,
        }, full_path)
        
        self.model_name = filename.replace('.pth', '')
        return full_path
        
    def save_checkpoint(self, path, episode_rewards, current_episode):
        """保存包含训练状态的完整检查点"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'current_episode': current_episode,
            'total_it': self.total_it,
        }, path)
        
    def load_checkpoint(self, path):
        """加载检查点并返回训练状态"""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        
        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if 'total_it' in checkpoint:
            self.total_it = checkpoint['total_it']
        
        episode_rewards = checkpoint.get('episode_rewards', [])
        current_episode = checkpoint.get('current_episode', 0)
        
        return episode_rewards, current_episode
        
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
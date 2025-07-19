## DDPG 算法定义的函数

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from my_nn import Actor, Critic, ReplayBuffer, Gru_Actor, Gru_Critic, Gru_ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
## DDPG 代理
class DDPGAgent:
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0, # state_dim 改为 1
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, sigma=0.2, clip_grad=False):
        """参数\n
        state_dim: 状态维度\n
        action_dim: 动作维度\n
        hidden_dim: 隐藏层维度\n
        action_bound: 动作范围\n
        actor_lr: Actor网络学习率\n
        critic_lr: Critic网络学习率\n
        gamma: 折扣因子\n
        tau: 软更新参数\n
        sigma: 探索噪声标准差\n
        clip_grad: 是否使用梯度裁剪\n
        """
        # 初始化参数
        self.gamma = gamma  # 折扣因子
        self.tau = tau      # 软更新参数
        self.sigma = sigma  # 探索噪声
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim # 保存 state_dim
        self.clip_grad = clip_grad # 是否使用梯度裁剪
        self.model_name = None # 当前加载的模型名称
        
        # 网络初始化
        self.actor = Actor(state_dim, action_dim, hidden_dim, action_bound).to(device)
        self.critic = Critic(state_dim, action_dim,  hidden_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim, action_bound).to(device)
        self.target_critic = Critic(state_dim, action_dim,  hidden_dim).to(device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器设置
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def select_action(self, state:np.ndarray, add_noise=True, epsilon=1.0, rand_prob=0.05, dt=None)-> np.ndarray:
        """选择动作，支持探索 (输入 state 是观测值)
        参数\n
        - state: 当前状态 (numpy array)\n
        - add_noise: 是否添加噪声 (布尔值)\n
        - epsilon: 噪声强度 (浮点数)\n
        - dt: 时间步长（为了兼容GRU版本，传统DDPG忽略此参数）\n
        返回：\n
        - action: 选择的动作 (numpy array)"""
        # state 应该是 numpy array, e.g., shape [action_dim,]
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # shape [1, state_dim]
        with torch.no_grad():
            action_tensor: torch.Tensor = self.actor(state)
            action_np: np.ndarray = action_tensor.cpu().detach().numpy()
            action = action_np.flatten()  # 展平 shape [action_dim,]
            
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.sigma * epsilon, size=self.action_dim)
            # print(noise)
            action += noise
            if np.random.random() < rand_prob:  # 有概率使用完全随机动作
                action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        # 返回单个动作值或动作数组
        return np.clip(action, -self.action_bound, self.action_bound)
    
    def update(self, replay_buffer:ReplayBuffer)-> Tuple[float, float]:
        """更新Actor和Critic网络\n"""
        if len(replay_buffer) < replay_buffer.batch_size:
            return 0.0, 0.0 # 返回 0 损失
        
        # 1. 从回放池中采样 (states 和 next_states 是观测值)
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        
        # 2. 计算目标 Q 值
        with torch.no_grad():
            next_actions: torch.Tensor = self.target_actor(next_states) # shape [batch_size, action_dim]
            target_q: torch.Tensor = self.target_critic(next_states, next_actions) # shape [batch_size, 1]
            target_value: torch.Tensor = rewards + self.gamma * target_q * (1 - dones) # shape [batch_size, 1]
            
        # 3. 更新Critic网络
        current_q: torch.Tensor = self.critic(states, actions) # shape [batch_size, 1]
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_grad: torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10) # 可选：梯度裁剪
        self.critic_optimizer.step()
        
        # 4. 更新Actor网络
        policy_actions: torch.Tensor = self.actor(states) # shape [batch_size, 1]
        actor_loss: torch.Tensor = -self.critic(states, policy_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_grad: torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10) # 可选：梯度裁剪
        self.actor_optimizer.step()
        
        # 5. 软更新目标网络
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source:Actor, target:Critic):
        """软更新目标网络参数"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
            
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)
        
    def save_checkpoint(self, path, episode_rewards, current_episode):
        """保存包含训练状态的完整检查点
        
        参数:
            path: 保存路径
            episode_rewards: 每轮的奖励记录
            current_episode: 当前训练轮次
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'current_episode': current_episode,
        }, path)
        
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        
    def load_checkpoint(self, path):
        """加载检查点并返回训练状态
        
        参数:
            path: 检查点路径
            
        返回:
            episode_rewards: 每轮的奖励记录
            current_episode: 当前训练轮次
        """
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        
        # 如果是完整的检查点（包含优化器状态和训练记录）
        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        episode_rewards = checkpoint.get('episode_rewards', [])
        current_episode = checkpoint.get('current_episode', 0)
        
        return episode_rewards, current_episode

## 基于GRU的DDPG代理
class GruDDPGAgent:
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0, 
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, sigma=0.2, 
                 clip_grad=False, seq_len=10, num_layers=2, use_time_input=False):
        """基于GRU的DDPG代理参数\n
        state_dim: 状态维度\n
        action_dim: 动作维度\n
        hidden_dim: 隐藏层维度\n
        action_bound: 动作范围\n
        actor_lr: Actor网络学习率\n
        critic_lr: Critic网络学习率\n
        gamma: 折扣因子\n
        tau: 软更新参数\n
        sigma: 探索噪声标准差\n
        clip_grad: 是否使用梯度裁剪\n
        seq_len: 序列长度\n
        num_layers: GRU层数\n
        use_time_input: 是否使用时间步长作为输入\n
        """
        # 初始化参数
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.clip_grad = clip_grad
        self.seq_len = seq_len
        self.use_time_input = use_time_input
        self.model_name = None
        
        # GRU网络初始化
        self.actor = Gru_Actor(state_dim, action_dim, hidden_dim, action_bound, seq_len, num_layers, use_time_input).to(device)
        self.critic = Gru_Critic(state_dim, action_dim, hidden_dim, seq_len, num_layers, use_time_input).to(device)
        self.target_actor = Gru_Actor(state_dim, action_dim, hidden_dim, action_bound, seq_len, num_layers, use_time_input).to(device)
        self.target_critic = Gru_Critic(state_dim, action_dim, hidden_dim, seq_len, num_layers, use_time_input).to(device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器设置
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 用于维护状态历史的缓冲区
        self.state_history = []
        self.time_history = [] if use_time_input else None
        
    def reset_state_history(self):
        """重置状态历史，在新的episode开始时调用"""
        self.state_history = []
        if self.time_history is not None:
            self.time_history = []
        
    def select_action(self, state: np.ndarray, add_noise=True, epsilon=1.0, rand_prob=0.05, dt=None) -> np.ndarray:
        """选择动作，支持探索
        参数\n
        - state: 当前状态 (numpy array)\n
        - add_noise: 是否添加噪声\n
        - epsilon: 噪声强度\n
        - rand_prob: 随机动作概率\n
        - dt: 当前时间步长（如果使用时间输入）\n
        返回：\n
        - action: 选择的动作 (numpy array)
        """
        # 更新状态历史
        self.state_history.append(state.copy())
        
        # 如果使用时间输入，更新时间历史
        if self.use_time_input and self.time_history is not None:
            if dt is not None:
                self.time_history.append(dt)
            else:
                # 如果没有提供时间步长，使用默认值
                self.time_history.append(0.001)
        
        # 如果历史长度不够，使用零填充或重复当前状态
        if len(self.state_history) < self.seq_len:
            # 用当前状态填充不足的部分
            padded_history = [state] * (self.seq_len - len(self.state_history)) + self.state_history
            if self.use_time_input and self.time_history is not None:
                current_dt = dt if dt is not None else 0.001
                padded_time_history = [current_dt] * (self.seq_len - len(self.time_history)) + self.time_history
        else:
            # 保持最近的seq_len个状态
            padded_history = self.state_history[-self.seq_len:]
            if self.use_time_input and self.time_history is not None:
                padded_time_history = self.time_history[-self.seq_len:]
            # 保持最近的seq_len个状态
            padded_history = self.state_history[-self.seq_len:]
            
        # 构建状态序列
        state_seq = np.array(padded_history)  # [seq_len, state_dim]
        state_seq_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_len, state_dim]
        
        # 如果使用时间输入，构建时间序列
        time_seq_tensor = None
        if self.use_time_input and self.time_history is not None:
            time_seq = np.array(padded_time_history)  # [seq_len]
            time_seq_tensor = torch.tensor(time_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, seq_len, 1]
        
        with torch.no_grad():
            if self.use_time_input:
                action_tensor = self.actor(state_seq_tensor, time_seq_tensor)
            else:
                # 为了兼容旧版本网络，传递None作为时间序列
                action_tensor = self.actor(state_seq_tensor, None)
            action_np = action_tensor.cpu().detach().numpy()
            action = action_np.flatten()
            
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.sigma * epsilon, size=self.action_dim)
            action += noise
            if np.random.random() < rand_prob:
                action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        return np.clip(action, -self.action_bound, self.action_bound)
    
    def update(self, replay_buffer: Gru_ReplayBuffer) -> Tuple[float, float]:
        """更新Actor和Critic网络"""
        if len(replay_buffer) < replay_buffer.batch_size:
            return 0.0, 0.0
        
        # 1. 从回放池中采样
        state_seqs, actions, rewards, next_state_seqs, dones, time_seqs, next_time_seqs = replay_buffer.sample()
        
        # 2. 计算目标Q值
        with torch.no_grad():
            if self.use_time_input:
                next_actions = self.target_actor(next_state_seqs, next_time_seqs)  # [batch_size, action_dim]
                target_q = self.target_critic(next_state_seqs, next_actions, next_time_seqs)  # [batch_size, 1]
            else:
                next_actions = self.target_actor(next_state_seqs, None)  # [batch_size, action_dim]
                target_q = self.target_critic(next_state_seqs, next_actions, None)  # [batch_size, 1]
            target_value = rewards + self.gamma * target_q * (1 - dones)  # [batch_size, 1]
            
        # 3. 更新Critic网络
        if self.use_time_input:
            current_q = self.critic(state_seqs, actions, time_seqs)  # [batch_size, 1]
        else:
            current_q = self.critic(state_seqs, actions, None)  # [batch_size, 1]
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_grad: 
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        self.critic_optimizer.step()
        
        # 4. 更新Actor网络
        if self.use_time_input:
            policy_actions = self.actor(state_seqs, time_seqs)  # [batch_size, action_dim]
            actor_loss = -self.critic(state_seqs, policy_actions, time_seqs).mean()
        else:
            policy_actions = self.actor(state_seqs, None)  # [batch_size, action_dim]
            actor_loss = -self.critic(state_seqs, policy_actions, None).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_grad: 
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        self.actor_optimizer.step()
        
        # 5. 软更新目标网络
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source, target):
        """软更新目标网络参数"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
            
    def save_model(self, save_path, episode):
        """保存模型"""
        import os
        from datetime import datetime
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gru_ddpg_model_ep{episode}_{current_time}.pth"
        full_path = os.path.join(save_path, filename)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': episode,
        }, full_path)
        
        self.model_name = filename.replace('.pth', '')
        return full_path
        
    def save_checkpoint(self, path, episode_rewards, current_episode):
        """保存包含训练状态的完整检查点"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'current_episode': current_episode,
        }, path)
        
    def load_checkpoint(self, path):
        """加载检查点并返回训练状态"""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        
        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        episode_rewards = checkpoint.get('episode_rewards', [])
        current_episode = checkpoint.get('current_episode', 0)
        
        return episode_rewards, current_episode
        
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
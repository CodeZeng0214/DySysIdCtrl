## DDPG 算法定义的函数

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Actor(nn.Module):
    """## Actor 网络\n
    策略网络，输出动作值\n
    ## 初始化参数\n
    - state_dim: 状态维度，默认值为 1
    - action_dim: 动作维度，默认值为 1
    - hidden_dim: 隐藏层维度，默认值为 64
    - action_bound: 动作范围，默认值为 5.0"""
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # 输出范围 [-1, 1]
        )
        self.action_bound = action_bound # 输出电流范围 [-action_bound, action_bound]
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state:torch.Tensor)-> torch.Tensor:
        action = self.net(state) * self.action_bound
        return action
    
## Critic 网络 - 价值网络
class Critic(nn.Module):
    """## Critic 网络\n
    价值网络，输出 Q 值\n
    ## 初始化参数\n
    - state_dim: 状态维度,默认值为 1
    - action_dim: 动作维度，默认值为 1
    - hidden_dim: 隐藏层维度，默认值为 64"""
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64): # state_dim 改为 1
        super(Critic, self).__init__()
        # 输入维度是 state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        # 最后一层使用较小的初始化
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)
        
    def forward(self, state:torch.Tensor, action:torch.Tensor)-> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        q_value = self.net(x)
        return q_value
    
## 经验回放池
class ReplayBuffer:
    """### 经验回放池，用于存储和采样经验\n
    ## 初始化参数\n
    - capacity: 回放池容量，默认值为 100000\n
    - batch_size: 采样批次大小，默认值为 64"""
    def __init__(self, capacity=100000, batch_size=64):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        
    def add(self, state:np.ndarray, action:np.ndarray, reward:float, next_state:np.ndarray, done:bool): # 添加 done 参数
        # state 和 next_state 是观测量，action 是动作值
        self.buffer.append((state, action, reward, next_state, done)) # 存储 done
        
    def sample(self)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # 返回类型添加 Tensor
        """随机采样一批经验\n
        返回：\n
        - states: 状态列表，shape [batch_size, state_dim]\n
        - actions: 动作列表，shape [batch_size, action_dim]\n
        - rewards: 奖励列表，shape [batch_size, 1]\n
        - next_states: 下一个状态列表，shape [batch_size, state_dim]\n
        - dones: 完成标志列表，shape [batch_size, 1]\n"""
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch) # 解包 dones

        # states 和 next_states 是观测值列表，每个元素是 shape [1,] 的 numpy array
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).reshape(-1, 1).to(device) # shape [batch_size, 1]
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).reshape(-1, 1).to(device) # shape [batch_size, 1]
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones, dtype=np.uint8), dtype=torch.float32).reshape(-1, 1).to(device) # shape [batch_size, 1]
        
        return states, actions, rewards, next_states, dones # 返回 dones
    
    def __len__(self):
        return len(self.buffer)
    
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
        
    def select_action(self, state:np.ndarray, add_noise=True, epsilon=1.0, rand_prob=0.05)-> np.ndarray:
        """选择动作，支持探索 (输入 state 是观测值)
        参数\n
        - state: 当前状态 (numpy array)\n
        - add_noise: 是否添加噪声 (布尔值)\n
        - epsilon: 噪声强度 (浮点数)\n
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
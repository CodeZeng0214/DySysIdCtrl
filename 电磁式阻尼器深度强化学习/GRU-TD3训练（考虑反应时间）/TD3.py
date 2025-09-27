## TD3 算法定义的函数

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, List
from nn import Actor, Critic, ReplayBuffer, Gru_Actor, Gru_Critic, Gru_ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## TD3代理基类
class BaseTD3Agent:
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, policy_sigma=0.2, clip_grad=False,
                 aware_dt: bool = False,
                 delay_enabled: bool = False, delay_step: int = 5, delay_sigma: int = 2, aware_delay_time: bool = False):
        """初始化TD3代理\n
        - state_dim 状态维度
        - action_dim 动作维度
        - hidden_dim 隐藏层维度
        - action_bound 动作范围
        - actor_lr Actor学习率
        - critic_lr Critic学习率
        - gamma 折扣因子
        - tau 软更新参数
        - policy_noise 目标策略平滑正则化噪声
        - noise_clip 噪声裁剪范围
        - policy_freq 策略更新频率
        - policy_sigma 探索噪声标准差
        - clip_grad 是否使用梯度裁剪
        - delay_enabled: 是否启用动作延迟
        - delay_step: 延迟步数
        - delay_sigma: 延迟步数的标准差
        """
        # 初始化参数
        self.state_dim = state_dim # 状态维度
        self.action_dim = action_dim # 动作维度
        self.hidden_dim = hidden_dim # 隐藏层维度
        self.action_bound = action_bound # 动作范围
        self.actor_lr = actor_lr # Actor网络学习率
        self.critic_lr = critic_lr # Critic网络学习率
        self.gamma = gamma # 折扣因子
        self.tau = tau # 软更新参数
        self.policy_noise = policy_noise # 目标策略平滑正则化噪声
        self.noise_clip = noise_clip # 噪声裁剪范围
        self.policy_freq = policy_freq # 策略更新频率
        self.policy_sigma = policy_sigma # 探索噪声标准差
        self.clip_grad = clip_grad # 是否使用梯度裁剪
        self.aware_dt = aware_dt # 是否使用时间步长作为状态的一部分
        self.delay_enabled = delay_enabled # 是否启用动作延迟
        self.delay_step = delay_step # 延迟步数
        self.delay_sigma = delay_sigma # 延迟步数的标准差
        self.aware_delay_time = aware_delay_time # 是否启用延迟感知

        self.model_name = None
        self.total_it = 0 # 总迭代次数
        self.episode_rewards = [] # 存储每个回合的奖励

    def _init_nn(self):
        # 需要在子类中定义
        self.actor: Actor | Gru_Actor = None
        self.critic1: Critic | Gru_Critic = None
        self.critic2: Critic | Gru_Critic = None
        self.target_actor: Actor | Gru_Actor = None
        self.target_critic1: Critic | Gru_Critic = None
        self.target_critic2: Critic | Gru_Critic = None
        raise NotImplementedError("需要在子类中初始化神经网络结构")

    def _init_optimizer(self):
        # 优化器设置
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def select_action(self):
        """选择动作"""
        raise NotImplementedError("需要在子类中实现动作选择方法")
            
    def reset_history(self):
        """重置状态历史，在新的episode开始时调用"""        
        if hasattr(self, 'state_history'):
            self.state_history = []
        
    def update(self, replay_buffer: Union[ReplayBuffer, Gru_ReplayBuffer]) -> Tuple[float, float, float]:
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
            
            # 打印梯度信息（调试用）
            total_grad_norm = 0
            param_count = 0
            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    total_grad_norm += grad_norm.item() ** 2
                    param_count += 1
                    if self.total_it % 1000 == 0:  # 每1000次打印一次
                        pass # 打印梯度信息
                        #print(f"  {name}: grad_norm={grad_norm:.6f}")
            total_grad_norm = total_grad_norm ** (1. / 2)
            if self.total_it % 1000 == 0:
                pass # 打印梯度信息
                #print(f"🔍 Actor总梯度范数: {total_grad_norm:.6f}, 参数数量: {param_count}")
            # 检查梯度是否爆炸
            if self.total_it % 1000 == 0 and total_grad_norm > 10:
                pass # 打印梯度信息
                #print(f"⚠️ 警告: Actor梯度过高! 梯度范数: {total_grad_norm}")
            # 检查梯度是否为零
            if self.total_it % 1000 == 0 and total_grad_norm < 1e-8:
                pass # 打印梯度信息
                #print(f"⚠️ 警告: Actor梯度几乎为零! 梯度范数: {total_grad_norm}")
            
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss, (critic1_loss.item() + critic2_loss.item()) / 2
    
## TD3 代理
class TD3Agent(BaseTD3Agent):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, sigma=0.2, clip_grad=False,
                 aware_dt: bool = False,
                 delay_enabled: bool = False, delay_step: int = 5, delay_sigma: int = 2, aware_delay_time: bool = False):
        # 初始化参数
        super().__init__(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, action_bound=action_bound,
                 actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau,
                 policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq, policy_sigma=sigma, clip_grad=clip_grad,
                 aware_dt=aware_dt,
                 delay_enabled=delay_enabled, delay_step=delay_step, delay_sigma=delay_sigma, aware_delay_time=aware_delay_time)
        self._init_nn()
        self._init_optimizer()

    def _init_nn(self):
        # 网络初始化
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_bound).to(device)
        self.critic1 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic2 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        
        self.target_actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_bound).to(device)
        self.target_critic1 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.target_critic2 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def select_action(self, state_history: List[np.ndarray], add_noise=True, epsilon=1.0, rand_prob=0.05, delay=1) -> float:
        """选择动作，支持探索"""
        # 如果启用动作延迟，使用延迟步数的高斯分布采样
        if self.delay_enabled:
            if len(state_history) < delay:
                state = state_history[-1]  # 如果历史长度不够，使用最新状态
            else:
                state = state_history[-delay]  # 使用延迟的状态
                
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_tensor: torch.Tensor = self.actor(state)
            action_np: np.ndarray = action_tensor.cpu().detach().numpy()
            action = action_np.flatten()
            
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.policy_sigma * epsilon, size=self.action_dim)
            action += noise
            if np.random.random() < rand_prob:
                action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        return float(np.clip(action, -self.action_bound, self.action_bound))
        
## 基于GRU的TD3代理
class Gru_TD3Agent(BaseTD3Agent):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, sigma=0.2, clip_grad=False, 
                 seq_len=10, gru_layers=1,
                 aware_dt: bool = False,
                 delay_enabled: bool = False, delay_step: int = 5, delay_sigma: int = 2, aware_delay_time: bool = False):
        self.seq_len = seq_len  # 序列长度
        self.gru_layers = gru_layers  # GRU层数
        super().__init__(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, action_bound=action_bound,
                         actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau,
                         policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq, policy_sigma=sigma, clip_grad=clip_grad,
                         aware_dt=aware_dt,
                         delay_enabled=delay_enabled, delay_step=delay_step, delay_sigma=delay_sigma, aware_delay_time=aware_delay_time)
        self._init_nn()
        self._init_optimizer()

    def _init_nn(self):
        # GRU网络初始化
        self.actor = Gru_Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_bound, self.seq_len, self.gru_layers).to(device)
        self.critic1 = Gru_Critic(self.state_dim, self.action_dim, self.hidden_dim, self.seq_len, self.gru_layers).to(device)
        self.critic2 = Gru_Critic(self.state_dim, self.action_dim, self.hidden_dim, self.seq_len, self.gru_layers).to(device)
        
        self.target_actor = Gru_Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_bound, self.seq_len, self.gru_layers).to(device)
        self.target_critic1 = Gru_Critic(self.state_dim, self.action_dim, self.hidden_dim, self.seq_len, self.gru_layers).to(device)
        self.target_critic2 = Gru_Critic(self.state_dim, self.action_dim, self.hidden_dim, self.seq_len, self.gru_layers).to(device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def select_action(self, state_history: List[np.ndarray], add_noise=True, epsilon=1.0, rand_prob=0.05, delay=1) -> float:
        """选择动作，支持探索"""

        # 如果历史长度不够，使用零填充或重复当前状态
        if len(state_history) < (self.seq_len + delay - 1):
            # 用当前状态填充不足的部分
            if len(state_history) < delay:
                # 如果历史长度连delay都不够，用起始状态填充
                padded_history = [state_history[0]] * (self.seq_len + delay - 1)
            else:
                # 修复delay=1时的切片问题
                if delay == 1:
                    padded_history = np.concatenate([[state_history[0]] * (self.seq_len - len(state_history)), state_history])
                else:
                    padded_history = np.concatenate([[state_history[0]] * ((self.seq_len + delay-1) - len(state_history)), state_history[:-(delay-1)]])
        else:
            # 保持最近的seq_len个状态
            if delay == 1:
                padded_history = state_history[-self.seq_len:]  # 当delay=1时，直接取最后seq_len个元素
            else:
                padded_history = state_history[-self.seq_len-(delay-1):-(delay-1)]        # 构建状态序列
        state_seq = np.array(padded_history)  # [seq_len, state_dim]
        state_seq_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_len, state_dim]

        with torch.no_grad():
            action_tensor: torch.Tensor = self.actor(state_seq_tensor)
            action_np: np.ndarray = action_tensor.cpu().detach().numpy()
            action = action_np.flatten()
            
        if add_noise:
            noise = np.random.normal(0, self.action_bound * self.policy_sigma * epsilon, size=self.action_dim)
            action += noise
            if np.random.random() < rand_prob:
                action = np.random.uniform(-self.action_bound, self.action_bound, self.action_dim)

        return float(np.clip(action, -self.action_bound, self.action_bound))
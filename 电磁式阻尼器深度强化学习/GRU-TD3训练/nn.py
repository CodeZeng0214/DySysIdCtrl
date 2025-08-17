import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gru_Actor(nn.Module):
    """## Gru_Actor 网络\n
    策略网络，输出动作值\n
    ## 初始化参数\n
    - state_dim: 状态维度，默认值为 1
    - action_dim: 动作维度，默认值为 1
    - hidden_dim: 隐藏层维度，默认值为 64
    - action_bound: 动作范围，默认值为 5.0
    - seq_len: 输入序列长度，默认值为 10
    """
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0, seq_len=10, num_layers=1):
        super(Gru_Actor, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.action_bound = action_bound
        
        # GRU层
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, 
                         num_layers=num_layers, batch_first=True, dropout=0.1)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        # 输出层使用小的均匀分布初始化
        nn.init.uniform_(self.output_layer[0, 2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer[0, 2].bias, -3e-3, 3e-3)

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        """前向传播
        Args:
            state_seq: 状态序列，shape [batch_size, seq_len, state_dim]
        Returns:
            action: 动作值，shape [batch_size, action_dim]
        """
        batch_size = state_seq.size(0)
        
        # GRU前向传播
        gru_out, _ = self.gru(state_seq)  # [batch_size, seq_len, hidden_dim]
        
        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 通过输出层得到动作
        action = self.output_layer(last_output) * self.action_bound
        
        return action
    
class Gru_Critic(nn.Module):
    """## Gru_Critic 网络\n
    价值网络，输出 Q 值\n
    ## 初始化参数\n
    - state_dim: 状态维度，默认值为 1
    - action_dim: 动作维度，默认值为 1
    - hidden_dim: 隐藏层维度，默认值为 64
    - seq_len: 输入序列长度，默认值为 10
    """
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, seq_len=10, num_layers=1):
        super(Gru_Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        # GRU层处理状态序列
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, 
                         num_layers=num_layers, batch_first=True, dropout=0.1)
        
        # 融合层：将GRU输出和动作结合
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
        for m in self.fusion_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # 最后一层使用较小的初始化
        nn.init.uniform_(self.fusion_layer[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fusion_layer[-1].bias, -3e-3, 3e-3)
        
    def forward(self, state_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播
        Args:
            state_seq: 状态序列，shape [batch_size, seq_len, state_dim]
            action: 动作值，shape [batch_size, action_dim]
        Returns:
            q_value: Q值，shape [batch_size, 1]
        """
        batch_size = state_seq.size(0)
        
        # GRU前向传播
        gru_out, _ = self.gru(state_seq)  # [batch_size, seq_len, hidden_dim]
        
        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 将状态特征和动作拼接
        x = torch.cat([last_output, action], dim=-1)  # [batch_size, hidden_dim + action_dim]
        
        # 通过融合层得到Q值
        q_value = self.fusion_layer(x)
        
        return q_value

class Gru_ReplayBuffer:
    """## Gru_ReplayBuffer 经验回放池\n
    用于存储和采样经验，支持序列状态\n
    ## 初始化参数\n
    - capacity: 回放池容量，默认值为 100000\n
    - batch_size: 采样批次大小，默认值为 64
    - seq_len: 序列长度，默认值为 10
    """
    def __init__(self, capacity=100000, batch_size=64, seq_len=10):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.state_history = deque(maxlen=seq_len)  # 用于维护状态历史
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """添加经验到回放池
        Args:
            state: 当前状态，shape [state_dim]
            action: 动作，shape [action_dim] 
            reward: 奖励
            next_state: 下一状态，shape [state_dim]
            done: 是否结束
        """
        # 更新状态历史
        self.state_history.append(state.copy())
        
        # 如果状态历史长度足够，则添加到经验池
        if len(self.state_history) >= self.seq_len:
            # 构建状态序列 [seq_len, state_dim]
            state_seq = np.array(list(self.state_history))
            
            # 构建下一状态序列（移除最旧的状态，添加next_state）
            next_state_seq = np.array(list(self.state_history)[1:] + [next_state])
            
            self.buffer.append((state_seq, action, reward, next_state_seq, done))
    
    def reset_history(self):
        """重置状态历史，在新的episode开始时调用"""
        self.state_history.clear()
        
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机采样一批经验\n
        返回：\n
        - state_seqs: 状态序列，shape [batch_size, seq_len, state_dim]\n
        - actions: 动作列表，shape [batch_size, action_dim]\n
        - rewards: 奖励列表，shape [batch_size, 1]\n
        - next_state_seqs: 下一状态序列，shape [batch_size, seq_len, state_dim]\n
        - dones: 完成标志列表，shape [batch_size, 1]\n
        """
        batch = random.sample(self.buffer, self.batch_size)
        state_seqs, actions, rewards, next_state_seqs, dones = zip(*batch)

        state_seqs = torch.tensor(np.array(state_seqs), dtype=torch.float32).to(device)  # [batch_size, seq_len, state_dim]
        actions = torch.tensor(np.array(actions), dtype=torch.float32).reshape(-1, 1).to(device)  # [batch_size, action_dim]
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).reshape(-1, 1).to(device)  # [batch_size, 1]
        next_state_seqs = torch.tensor(np.array(next_state_seqs), dtype=torch.float32).to(device)  # [batch_size, seq_len, state_dim]
        dones = torch.tensor(np.array(dones, dtype=np.uint8), dtype=torch.float32).reshape(-1, 1).to(device)  # [batch_size, 1]
        
        return state_seqs, actions, rewards, next_state_seqs, dones
    
    def __len__(self):
        return len(self.buffer)
    
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
        # 输出层使用小的均匀分布初始化
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.constant_(self.net[-2].bias, 0)

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
        nn.init.constant_(self.net[-1].bias, 0)

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
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from typing import Tuple
import torch.nn.functional as F  # 添加导入

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GruPredictor(nn.Module):
    """独立的GRU预测网络，用于序列预测
    用于在线训练，预测未来的状态序列
    
    ## 初始化参数
    - state_dim: 部分状态维度，默认值为 2
    - hidden_dim: GRU隐藏层维度，默认值为 64
    - num_layers: GRU层数，默认值为 1
    - pre_seq_len: 预测的未来时间步数，默认值为 5
    """
    def __init__(self, state_dim=2, hidden_dim=64, num_layers=1, fc_seq_len=5,
                 aware_dt=False, aware_delay_time=False):
        super(GruPredictor, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc_seq_len = fc_seq_len
        self.aware_dt = aware_dt
        self.aware_delay_time = aware_delay_time
        
        # GRU层
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, 
                         num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        
        # 用于预测下一个状态的线性层
        self.fc_predict = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, state_dim))
        
        # 添加注意力层（用于处理预测的状态序列）
        self.embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2))
        self.attention = nn.Sequential(nn.Linear(state_dim+4+hidden_dim//2, 1))  # 计算每个时间步的注意力分数
        
        self._init_weights()

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        """前向传播
        Args:
            state_seq: 输入状态序列，shape [batch_size, seq_len, state_dim]
        Returns:
            state_vector: 注意力加权后的状态向量，shape [batch_size, state_dim]
        """
        batch_size = state_seq.size(0)
        
        # 处理输入状态
        if state_seq.size(2) == self.state_dim:
            gru_state_seq = state_seq
        else:
            gru_state_seq = self.del_vel_acc(state_seq)  # 删除速度和加速度维度

        # GRU处理输入序列
        gru_out, h_n = self.gru(gru_state_seq)  # gru_out: [batch_size, seq_len, hidden_dim]
                                            # h_n: [num_layers, batch_size, hidden_dim]
        
        # 使用最后的隐藏状态开始预测未来序列
        predicted_states = []
        hidden_states = []
        
        # 预测第一个未来状态
        current_h = h_n  # [num_layers, batch_size, hidden_dim]
        current_state = self.fc_predict(current_h[-1]).unsqueeze(1)  # [batch_size, 1, state_dim]

        for t in range(self.fc_seq_len):
            predicted_states.append(current_state)
            hidden_states.append(current_h[-1].unsqueeze(1))  # [batch_size, 1, hidden_dim]

            # 使用预测的状态作为下一步输入
            _, next_h = self.gru(current_state, current_h)
            next_state = self.fc_predict(next_h[-1]).unsqueeze(1)
            
            current_state, current_h = next_state, next_h

        state_fc_seq = torch.cat(predicted_states, dim=1)  # [batch_size, fc_seq_len, state_dim]
        h_fc_seq = torch.cat(hidden_states, dim=1)  # [batch_size, fc_seq_len, hidden_dim]
        
        # 计算速度和加速度，拼接到状态序列
        vel_fc_seq, acc_fc_seq = self.compute_derivatives(state_fc_seq[:,:,0:2],state_fc_seq[:,:,2] if self.aware_dt else None)
        state_fc_seq = torch.cat((state_fc_seq[:,:,0].unsqueeze(2), vel_fc_seq[:,:,0].unsqueeze(2), acc_fc_seq[:,:,0].unsqueeze(2), 
                                    state_fc_seq[:,:,1].unsqueeze(2), vel_fc_seq[:,:,1].unsqueeze(2), acc_fc_seq[:,:,1].unsqueeze(2), 
                                    state_fc_seq[:,:,2:]), dim=2)  # [batch_size, fc_seq_len, state_dim]

        state_combined_seq = torch.cat((state_fc_seq[:, -self.fc_seq_len//2:, :], state_fc_seq), dim=1)  # [batch_size, 1.5*fc_seq_len, state_dim]
        h_combined_seq = torch.cat((gru_out[:, -self.fc_seq_len//2:, :], h_fc_seq), dim=1)  # [batch_size, 1.5*fc_seq_len, hidden_dim]
        
        # 计算注意力权重
        attention_scores = self.attention(torch.cat((self.embedding(h_combined_seq), state_combined_seq), dim=2)).squeeze(-1)  # [batch_size, 1.5*fc_seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1.5*fc_seq_len]

        # 加权求和得到上下文状态向量
        # [batch_size, 1.5*fc_seq_len, state_dim] * [batch_size, 1.5*fc_seq_len, 1] -> [batch_size, 1.5*fc_seq_len, state_dim]
        state_vector = torch.sum(state_combined_seq * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, state_dim]

        return state_vector
    
    def del_vel_acc(self, state_seq: torch.Tensor) -> torch.Tensor:
        """删除状态序列中的速度和加速度维度，返回处理后的状态序列
        - 注意state_seq的shape为 [batch_size, seq_len, full_state_dim]"""
        if self.aware_dt or self.aware_delay_time:  # 判断是否包含感知状态
           gru_state_seq = torch.cat((state_seq[:,:,0].unsqueeze(2),state_seq[:,:,3].unsqueeze(2),state_seq[:,:,6:]), dim=2) # 处理状态序列，抛弃速度和加速度维度
        else:
            gru_state_seq = torch.cat((state_seq[:,:,0].unsqueeze(2),state_seq[:,:,3].unsqueeze(2)), dim=2) # 处理状态序列，抛弃速度和加速度维度
        return gru_state_seq
        
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for m in self.fc_predict.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def unfreeze_gru(self):
        """解冻GRU预测器，用于微调"""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_gru(self):
        """冻结GRU预测器"""
        for param in self.parameters():
            param.requires_grad = False
            
    # def predict_states(self, state_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """使用GRU预测未来状态序列，输出拼接后的状态序列和隐藏状态序列"""
    #     # 使用共享的GRU预测器获取预测序列
    #     # with torch.no_grad():  # GRU预测器的输出不参与梯度计算
    #     if self.aware_dt or self.aware_delay_time:  # 判断是否包含感知状态
    #         gru_state_seq = torch.cat((state_seq[:,:,0].unsqueeze(2),state_seq[:,:,3].unsqueeze(2),state_seq[:,:,6:]), dim=2) # 处理状态序列，抛弃速度和加速度维度
    #     else:
    #         gru_state_seq = torch.cat((state_seq[:,:,0].unsqueeze(2),state_seq[:,:,3].unsqueeze(2)), dim=2) # 处理状态序列，抛弃速度和加速度维度

    #     gru_state_fc_seq, h_fc_seq = self.forward(gru_state_seq)  # [batch_size, fc_seq_len, state_dim], [batch_size, fc_seq_len, hidden_dim]

    #     # 计算速度和加速度，拼接到状态序列
    #     vel_fc_seq, acc_fc_seq = self.compute_derivatives(gru_state_fc_seq[:,:,0:2],gru_state_fc_seq[:,:,2] if self.aware_dt else None)
    #     state_fc_seq = torch.cat((gru_state_fc_seq[:,:,0].unsqueeze(2), vel_fc_seq[:,:,0].unsqueeze(2), acc_fc_seq[:,:,0].unsqueeze(2), 
    #                                 gru_state_fc_seq[:,:,1].unsqueeze(2), vel_fc_seq[:,:,1].unsqueeze(2), acc_fc_seq[:,:,1].unsqueeze(2), 
    #                                 gru_state_fc_seq[:,:,2:]), dim=2)  # [batch_size, fc_seq_len, state_dim]

    #     state_combined_seq = torch.cat((state_seq[:, -self.fc_seq_len//2:, :], state_fc_seq), dim=1)  # [batch_size, 1.5*fc_seq_len, state_dim]
    #     h_combined_seq = torch.cat((self.gru_out[:, -self.fc_seq_len//2:, :], h_fc_seq), dim=1)  # [batch_size, 1.5*fc_seq_len, hidden_dim]
        
    #     # 计算注意力权重
    #     attention_scores = self.attention(torch.cat((self.embedding(h_combined_seq), state_combined_seq), dim=2)).squeeze(-1)  # [batch_size, 1.5*fc_seq_len]
    #     attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1.5*fc_seq_len]

    #     # 加权求和得到上下文状态向量
    #     state_vector = torch.sum(state_combined_seq * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, state_dim]

    #     return state_vector
    
    def compute_derivatives(self, positions: torch.Tensor, dt_seq: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """通过数值微分计算速度和加速度（改进版）
        
        使用策略：
        - 速度：中心差分（内部） + 线性外推（边界）
        - 加速度：基于速度的中心差分 + 外推
        
        Args:
            positions: 位移序列 [batch_size, seq_len, position_dim]
            dt_seq: 时间步长序列 [batch_size, seq_len] (可选)
        
        Returns:
            velocities: 速度序列 [batch_size, seq_len, position_dim]
            accelerations: 加速度序列 [batch_size, seq_len, position_dim]
        """
        batch_size, seq_len, pos_dim = positions.shape
        
        # 如果没有提供时间步长，使用默认值
        if dt_seq is None:
            dt = 0.001
            dt_tensor = torch.full((batch_size, seq_len), dt, device=positions.device)
        else:
            dt_tensor = dt_seq
            dt = dt_tensor.mean().item()
        
        # ========== 方法1：中心差分 + 边界外推（推荐用于速度） ==========
        velocities = torch.zeros_like(positions)
        
        if seq_len >= 3:
            # 内部点：使用中心差分 v(i) = [x(i+1) - x(i-1)] / (2*dt)
            # 精度: O(dt^2)
            for i in range(1, seq_len - 1):
                dt_avg = (dt_tensor[:, i-1] + dt_tensor[:, i]) / 2
                velocities[:, i, :] = (positions[:, i+1, :] - positions[:, i-1, :]) / (2 * dt_avg.unsqueeze(-1))
            
            # 起始点：前向差分 v(0) = [x(1) - x(0)] / dt
            velocities[:, 0, :] = (positions[:, 1, :] - positions[:, 0, :]) / dt_tensor[:, 0].unsqueeze(-1)
            
            # 终止点：后向差分 v(n) = [x(n) - x(n-1)] / dt
            velocities[:, -1, :] = (positions[:, -1, :] - positions[:, -2, :]) / dt_tensor[:, -1].unsqueeze(-1)
            
        elif seq_len == 2:
            # 序列太短，只能用前向差分
            velocities[:, 0, :] = (positions[:, 1, :] - positions[:, 0, :]) / dt_tensor[:, 0].unsqueeze(-1)
            velocities[:, 1, :] = velocities[:, 0, :]
        else:
            # 只有一个点，速度为0
            velocities[:, 0, :] = 0
        
        # ========== 加速度：基于速度序列的中心差分 ==========
        accelerations = torch.zeros_like(positions)
        
        if seq_len >= 3:
            # 内部点：中心差分 a(i) = [v(i+1) - v(i-1)] / (2*dt)
            for i in range(1, seq_len - 1):
                dt_avg = (dt_tensor[:, i-1] + dt_tensor[:, i]) / 2
                accelerations[:, i, :] = (velocities[:, i+1, :] - velocities[:, i-1, :]) / (2 * dt_avg.unsqueeze(-1))
            
            # 起始点：前向差分
            accelerations[:, 0, :] = (velocities[:, 1, :] - velocities[:, 0, :]) / dt_tensor[:, 0].unsqueeze(-1)
            
            # 终止点：后向差分
            accelerations[:, -1, :] = (velocities[:, -1, :] - velocities[:, -2, :]) / dt_tensor[:, -1].unsqueeze(-1)
            
        elif seq_len == 2:
            accelerations[:, 0, :] = (velocities[:, 1, :] - velocities[:, 0, :]) / dt_tensor[:, 0].unsqueeze(-1)
            accelerations[:, 1, :] = accelerations[:, 0, :]
        else:
            accelerations[:, 0, :] = 0
        
        return velocities, accelerations

class Gru_Actor(nn.Module):
    """## Gru_Actor 网络（分离式架构）\n
    策略网络，使用共享的GRU预测器，然后通过注意力层和全连接层输出动作值\n
    ## 初始化参数\n
    - state_dim: 状态维度，默认值为 1
    - action_dim: 动作维度，默认值为 1
    - hidden_dim: 隐藏层维度，默认值为 64
    - action_bound: 动作范围，默认值为 5.0
    - seq_len: 输入序列长度，默认值为 10
    - gru_predictor: 共享的GRU预测器（必需参数）
    """
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0):
        super(Gru_Actor, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_bound = action_bound
        self.state_dim = state_dim
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        # 输出层使用小的均匀分布初始化
        nn.init.uniform_(self.output_layer[2].weight, -3e-3, 3e-3)
        nn.init.constant_(self.output_layer[2].bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播
        Args:
            state: 状态向量，shape [batch_size, state_dim]
        Returns:
            action: 动作值，shape [batch_size, action_dim]
        """

        batch_size = state.size(0)

        # 通过输出层得到动作
        action = self.output_layer(state) * self.action_bound

        return action
    
class Gru_Critic(nn.Module):
    """## Gru_Critic 网络（分离式架构）\n
    价值网络，使用共享的GRU预测器，然后通过注意力层和全连接层输出 Q 值\n
    ## 初始化参数\n
    - state_dim: 状态维度，默认值为 1
    - action_dim: 动作维度，默认值为 1
    - hidden_dim: 隐藏层维度，默认值为 64
    - seq_len: 输入序列长度，默认值为 10
    - gru_predictor: 共享的GRU预测器（必需参数）
    """
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=64):
        super(Gru_Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # 融合层：将注意力加权后的状态特征和动作结合
        self.fusion_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.fusion_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # 输出层使用小的均匀分布初始化
        nn.init.uniform_(self.fusion_layer[-1].weight, -3e-3, 3e-3)
        nn.init.constant_(self.fusion_layer[-1].bias, 0)

    def forward(self, state: Tuple[torch.Tensor, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """前向传播
        Args:
            state: 状态向量，shape [batch_size, state_dim]
            action: 动作值，shape [batch_size, action_dim]
        Returns:
            q_value: Q值，shape [batch_size, 1]
        """
        # 将状态特征和动作拼接
        x = torch.cat([state, action], dim=-1)  # [batch_size, state_dim + action_dim]

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
        
    def add(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool):
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
            
    def delete_last(self, num):
        """删除最近添加的 num 条经验"""
        for _ in range(num):
            if self.buffer:
                self.buffer.pop()
    
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
        # print(state_seqs.shape)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).reshape(-1, 1).to(device)  # [batch_size, action_dim]
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).reshape(-1, 1).to(device)  # [batch_size, 1]
        next_state_seqs = torch.tensor(np.array(next_state_seqs), dtype=torch.float32).to(device)  # [batch_size, seq_len, state_dim]
        dones = torch.tensor(np.array(dones, dtype=np.uint8), dtype=torch.float32).reshape(-1, 1).to(device)  # [batch_size, 1]
        
        return state_seqs, actions, rewards, next_state_seqs, dones
    
    def __len__(self):
        return len(self.buffer)

class GruPredictorBuffer:
    """## GRU预测器专用回放池
    用于训练GRU预测器，存储时延状态序列和对应的真实未来状态
    
    ## 初始化参数
    - capacity: 回放池容量
    - batch_size: 批次大小
    - seq_len: 输入序列长度（时延状态序列）
    - pre_seq_len: 预测的未来时间步数
    """
    def __init__(self, capacity=100000, batch_size=64, seq_len=10, fc_seq_len=5,
                 aware_dt=False, aware_delay_time=False, dt=0.001):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.fc_seq_len = fc_seq_len
        self.aware_dt = aware_dt
        self.aware_delay_time = aware_delay_time
        self.dt = dt  # 默认时间步长

    def add_from_full_history(self, full_state_history: list):
        """从完整的状态历史中提取训练样本
        Args:
            full_state_history: 完整的状态历史列表（无时延的真实状态）
        """
        # 需要至少 self.seq_len + self.fc_seq_len 个状态
        min_length = self.seq_len + self.fc_seq_len
        
        if len(full_state_history) < min_length:
            return
        fc_seq = np.array(full_state_history[-self.fc_seq_len:])
        pre_seq = np.array(full_state_history[-(self.seq_len + self.fc_seq_len):-self.fc_seq_len])
        
        # 处理状态序列，抛弃速度和加速度维度
        pre_seq = np.delete(pre_seq, [1, 2, 4, 5], axis=1)
        fc_seq = np.delete(fc_seq, [1, 2, 4, 5], axis=1)
        
        # 根据时延信息获取应该输出的真实状态
        if self.aware_dt: fc_dt_seq = np.concatenate([pre_seq[-1, 2], fc_seq[:, 2]])
        else: fc_dt_seq = np.full((self.fc_seq_len+1,), self.dt)
        if self.aware_delay_time: fc_delay_time = pre_seq[-1, 2+int(self.aware_dt)] 
        else: fc_delay_time = self.dt
        time_diffs = np.abs(np.cumsum(fc_dt_seq) - fc_delay_time)  # [fc_seq_len+1,]
        fc_index = np.argmin(time_diffs)
        fc_state = fc_seq[fc_index]

        self.buffer.append((
            np.array(pre_seq),      # [seq_len, state_dim]
            np.array(fc_state)   # [fc_seq_len, state_dim]
        ))

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机采样一批训练数据
        Returns:
            - pre_seqs: 时延输入序列，shape [batch_size, seq_len, state_dim]
            - fc_state: 真实的时延后状态，shape [batch_size, state_dim]
        """
        batch = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))
        pre_seqs, fc_state = zip(*batch)

        pre_seqs = torch.tensor(np.array(pre_seqs), dtype=torch.float32).to(device)
        fc_state = torch.tensor(np.array(fc_state), dtype=torch.float32).to(device)

        return pre_seqs, fc_state
    
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

    def add(self, state:np.ndarray, action:float, reward:float, next_state:np.ndarray, done:bool):
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))

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
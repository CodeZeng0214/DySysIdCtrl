import copy
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mlp_Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_bound: float) -> None:
        super().__init__()
        self.action_bound = action_bound
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.action_bound


class Mlp_Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class GRUEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int, use_attention: bool = True) -> None:
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(state_seq)
        if not self.use_attention:
            return out[:, -1]
        attn_score = self.attn(out).squeeze(-1)
        weight = torch.softmax(attn_score, dim=-1)
        return torch.sum(out * weight.unsqueeze(-1), dim=1)
    
class GruPredictor(nn.Module):
    """独立的GRU预测网络，用于序列预测\n
    采用归一化架构
        
    ## 初始化参数
    - state_dim: 全部状态维度，默认值为 6
    - hidden_dim: GRU隐藏层维度，默认值为 64
    - num_layers: GRU层数，默认值为 1
    - pre_seq_len: 预测的未来时间步数，默认值为 5
    """
    def __init__(self, norm=False, simple_nn=False,freeze_gru=False,
                 state_dim=6, hidden_dim=64, num_layers=1, fc_seq_len=5):
        super(GruPredictor, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc_seq_len = fc_seq_len
        self.simple_nn = simple_nn

        # GRU层
        self.gru_norm = nn.LayerNorm(state_dim) if norm else nn.Identity()
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim,
                                         num_layers=num_layers, batch_first=True, dropout=0.1)

        # 用于预测下一个状态的线性层
        self.fc_predict = nn.Sequential(nn.Linear(hidden_dim, state_dim))
        
        # 添加注意力层（用于处理预测的状态序列）
        self.attention_norm = nn.LayerNorm(state_dim) if norm else nn.Identity()
        self.embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 1))  # 计算每个时间步的注意力分数

        self._init_weights()
        
        # 冻结GRU参数
        if freeze_gru:
            self.freeze_gru()

    def forward(self, state_seq: torch.Tensor, critic: bool=False) -> torch.Tensor:
        """前向传播
        Args:
            state_seq: 输入状态序列，shape [batch_size, seq_len, state_dim]
        Returns:
            state_vector: 注意力加权后的状态向量，shape [batch_size, state_dim]
        """
        # GRU处理输入序列
        gru_out, h_n = self.gru(self.gru_norm(state_seq))  # gru_out: [batch_size, seq_len, hidden_dim]
                                            # h_n: [num_layers, batch_size, hidden_dim]
        
        if not self.simple_nn:
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
            
            state_combined_seq = torch.cat((state_seq[:, -self.fc_seq_len//2:, :], state_fc_seq), dim=1)  # [batch_size, 1.5*fc_seq_len, state_dim]
            h_combined_seq = torch.cat((gru_out[:, -self.fc_seq_len//2:, :], h_fc_seq), dim=1)  # [batch_size, 1.5*fc_seq_len, hidden_dim]
            
            # 计算注意力权重
            # attention_scores = self.attention(torch.cat((h_combined_seq, self.attention_norm(state_combined_seq)), dim=2)).squeeze(-1)  # [batch_size, 1.5*fc_seq_len]
            attention_scores = self.attention(h_combined_seq).squeeze(-1)  # [batch_size, 1.5*fc_seq_len]
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1.5*fc_seq_len]

            # 加权求和得到上下文状态向量
            # [batch_size, 1.5*fc_seq_len, state_dim] * [batch_size, 1.5*fc_seq_len, 1] -> [batch_size, 1.5*fc_seq_len, state_dim]
            state_vector = torch.sum(state_combined_seq * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, state_dim]
        else:
            # 简单模式下，直接使用GRU的输出
            if critic:
                state_vector = h_n[-1]
            else:
                # 直接使用最后一个时间步的输出作为状态向量
                state_vector = self.fc_predict(h_n[-1])  # [batch_size, state_dim]
        return state_vector
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name: ###
                nn.init.orthogonal_(param) ###
            # if 'weight_ih' in name:  # 输入到隐藏层的权重
            #     nn.init.xavier_uniform_(param)
            # elif 'weight_hh' in name:  # 隐藏层到隐藏层的权重
            #     nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for m in self.fc_predict.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight) ###
                nn.init.constant_(m.bias, 0)
        
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def unfreeze_all(self):
        """解冻GRU预测器，用于微调"""
        for param in self.parameters():
            param.requires_grad = True
    
    def unfreeze_gru(self):
        """解冻GRU预测器，用于微调"""
        for param in self.gru.parameters():
            param.requires_grad = True
    
    def freeze_all(self):
        """冻结GRU预测器"""
        for param in self.parameters():
            param.requires_grad = False
            
    def freeze_gru(self):
        """冻结GRU预测器"""
        for param in self.gru.parameters():
            param.requires_grad = False

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
    def __init__(self, gru_predictor: GruPredictor, norm=False, simple_nn=False,
                 state_dim=1, action_dim=1, hidden_dim=64, action_bound=5.0):
        super(Gru_Actor, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_bound = action_bound
        self.state_dim = state_dim
        self.gru_predictor = gru_predictor

        # 输出层
        self.output_layer = nn.Sequential(
            # nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # for m in self.output_layer.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)
        # # 输出层使用Xavier 初始化
        # nn.init.xavier_uniform_(self.output_layer[-2].weight)
        
        ### 
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0) ###
        ### 输出层使用小的均匀分布初始化 ### 
        nn.init.uniform_(self.output_layer[0].weight, -3e-3, 3e-3) ###
        nn.init.constant_(self.output_layer[0].bias, 0) ### 
        nn.init.uniform_(self.output_layer[-2].weight, -3e-3, 3e-3) ###
        nn.init.constant_(self.output_layer[-2].bias, 0) ###

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播
        Args:
            state: 状态向量，shape [batch_size, state_dim]
        Returns:
            action: 动作值，shape [batch_size, action_dim]
        """

        # 通过GRU预测器得到注意力加权后的状态向量
        state = self.gru_predictor(state)  # [batch_size, state_dim]

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
    def __init__(self, gru_predictor: GruPredictor, norm=False, simple_nn=False,
                 state_dim=1, action_dim=1, hidden_dim=64, gru_hidden_dim=64):
        super(Gru_Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gru_predictor = gru_predictor

        self.state_norm = nn.LayerNorm(gru_hidden_dim) if norm else nn.Identity()
        # 融合层：将注意力加权后的状态特征和动作结合
        self.fusion_layer = nn.Sequential(
            # nn.LayerNorm(gru_hidden_dim + action_dim),
            # nn.Linear(gru_hidden_dim + action_dim, hidden_dim),
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # for m in self.fusion_layer.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)
        
        ###
        for m in self.fusion_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        # 最后一层使用较小的初始化
        nn.init.uniform_(self.fusion_layer[-1].weight, -3e-3, 3e-3)
        nn.init.constant_(self.fusion_layer[-1].bias, 0) ###

    def forward(self, state: Tuple[torch.Tensor, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """前向传播
        Args:
            state: 状态向量，shape [batch_size, state_dim]
            action: 动作值，shape [batch_size, action_dim]
        Returns:
            q_value: Q值，shape [batch_size, 1]
        """
        # state 通过 GRU 预测器得到注意力加权后的状态向量
        # state = self.gru_predictor(state)  # [batch_size, state_dim]
        
        hidden_state = self.gru_predictor(state, critic=True)  # [batch_size, hidden_dim]
        
        # 将状态特征和动作拼接
        x = torch.cat([self.state_norm(hidden_state), action], dim=-1)  # [batch_size, state_dim + action_dim]

        # 通过融合层得到Q值
        q_value = self.fusion_layer(x)
        
        return q_value

def build_actor_critic(
    arch: str,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    action_bound: float,
    gru_hidden: int = 64,
    gru_layers: int = 1,
    fc_seq_len: int = 5,
    norm: bool = False, # 是否使用归一化
    simple_nn: bool = False, # 是否使用简单神经网络
) -> Tuple[Mlp_Actor|Gru_Actor, Mlp_Critic|Gru_Critic]:
    arch = arch.lower()
    if arch == "mlp":
        actor = Mlp_Actor(state_dim, action_dim, hidden_dim, action_bound)
        critic = Mlp_Critic(state_dim, action_dim, hidden_dim)
    elif arch == "seq":
        gru_predictor1 = GruPredictor(norm=norm, simple_nn=simple_nn, 
                                     state_dim=state_dim, hidden_dim=gru_hidden, 
                                     num_layers=gru_layers, fc_seq_len=fc_seq_len)
        gru_predictor2 = GruPredictor(norm=norm, simple_nn=simple_nn, 
                                     state_dim=state_dim, hidden_dim=gru_hidden, 
                                     num_layers=gru_layers, fc_seq_len=fc_seq_len)
        actor = Gru_Actor(gru_predictor1, norm=norm, simple_nn=simple_nn,
                            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        critic = Gru_Critic(gru_predictor2, norm=norm, simple_nn=simple_nn,
                            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return actor.to(device), critic.to(device)

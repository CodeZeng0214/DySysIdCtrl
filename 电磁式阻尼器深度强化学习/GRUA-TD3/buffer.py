import numpy as np
import torch
from typing import Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """经验回放池，支持按序列采样和动作延迟处理。"""

    def __init__(self, state_dim: int, capacity: int = 100000, seq_len: int = 1) -> None:
        self.state_dim = state_dim # 状态维度
        self.capacity = capacity # 最大容量
        self.seq_len = max(1, seq_len) # 取样的目标序列长度，至少为1
        self.states = np.zeros((capacity, state_dim), dtype=np.float32) # 状态数组
        self.actions = np.zeros((capacity, 1), dtype=np.float32) # 动作数组
        self.rewards = np.zeros((capacity, 1), dtype=np.float32) # 奖励数组 
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32) # 下一个状态数组
        self.dones = np.zeros((capacity, 1), dtype=np.float32) # 结束标志数组
        self.delay_steps = np.zeros((capacity, 1), dtype=np.int32) # 延迟步数数组，对应每个样本的动作延迟信息
        self.ptr = 0 # 当前写入位置
        self.size = 0 # 当前存储的样本数量

    def add(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool, delay: int = 0) -> None:
        """添加一个新的样本到经验池。"""
        idx = self.ptr % self.capacity # 环形经验池写入位置，覆盖旧数据
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.delay_steps[idx] = int(delay)
        self.ptr = (self.ptr + 1) % self.capacity # 更新写入位置，模拟队列
        self.size = min(self.size + 1, self.capacity) # 更新当前样本数量，最多为容量

    def __len__(self) -> int:
        return self.size

    def _chronological_views(self):
        """返回按时间顺序排列的各字段视图，用于序列采样。"""
        base = (self.ptr - self.size) % self.capacity # 计算当前数据的起始位置，保证时间顺序
        order = (base + np.arange(self.size)) % self.capacity # 按时间顺序的索引数组
        return (
            self.states[order],
            self.actions[order],
            self.rewards[order],
            self.next_states[order],
            self.dones[order],
            self.delay_steps[order],
        )
    
    def sample(self, batch_size: int, use_sequence: bool = False, apply_delay: bool = True) -> Tuple[torch.Tensor, ...]:
        """采样一个批次的数据。\n
        如果 use_sequence 为 True，则采样序列数据，返回形状为 (batch_size, seq_len, state_dim) 的状态序列等。\n
        如果 use_sequence 为 False，则采样单步数据，返回形状为 (batch_size, state_dim) 的状态等。\n
        如果 apply_delay 为 True，则在采样时考虑动作延迟，确保采样的索引不会导致越界。"""
        assert self.size >= batch_size, "Not enough samples" # 确保有足够样本

        # 获取按时间顺序排列的各字段视图
        states_arr, actions_arr, rewards_arr, next_states_arr, dones_arr, delays_arr = self._chronological_views()

        idxs: np.ndarray = self._sample_indices(batch_size, use_sequence, apply_delay, dones_arr, delays_arr)

        # 根据采样的索引提取数据
        if not use_sequence or self.seq_len == 1:
            # 非序列采样，直接提取对应索引的数据
            assert idxs.ndim == 1, "非序列抽样情况下，索引应为一维数组"
            states = torch.tensor(states_arr[idxs], device=device)
            actions = torch.tensor(actions_arr[idxs], device=device)
            rewards = torch.tensor(rewards_arr[idxs], device=device)
            next_states = torch.tensor(next_states_arr[idxs], device=device)
            dones = torch.tensor(dones_arr[idxs], device=device)
            return states, actions, rewards, next_states, dones
        elif use_sequence and self.seq_len > 1:
            # 序列采样，提取对应索引范围内的数据序列
            assert idxs.ndim == 2, "序列抽样情况下，索引应为二维数组"
            idx_seqs = idxs
            state_seqs = torch.tensor(states_arr[idx_seqs[:,0]:idx_seqs[:,1], :], device=device) # 序列状态数据，形状为 (batch_size, seq_len, state_dim)
            action_seqs = torch.tensor(actions_arr[idx_seqs[:,1], :], device=device) # 动作数据，形状为 (batch_size, action_dim)
            reward_seqs = torch.tensor(rewards_arr[idx_seqs[:,1], :], device=device) # 奖励数据，形状为 (batch_size, 1)
            next_state_seqs = torch.tensor(next_states_arr[idx_seqs[:,1], :], device=device) # 下一个状态数据，形状为 (batch_size, state_dim)
            done_seqs = torch.tensor(dones_arr[idx_seqs[:,1], :], device=device) # 结束标志数据，形状为 (batch_size, 1)
            return state_seqs, action_seqs, reward_seqs, next_state_seqs, done_seqs            
                
    def reset(self) -> None:
        self.ptr = 0
        self.size = 0

    def _sample_indices(self, batch_size: int, use_sequence: bool, apply_delay: bool,
                       dones_arr: np.ndarray, delays_arr: np.ndarray) -> np.ndarray:
        """采样符合条件的索引，用于后续数据提取。"""
        # 非序列采样，直接随机采样单步数据
        if not use_sequence or self.seq_len == 1:
            idxs = np.random.randint(0, self.size, size=batch_size*2) # 多采样以备筛选
            if apply_delay:
                # 如果应用延迟，确保采样的索引不会导致越界
                idxs = np.array([i - delays_arr[i, 0] for i in idxs if i - delays_arr[i, 0] >= 0]) # 调整索引以考虑延迟

        # 序列采样，基于上面得到的延迟索引进行筛选
        idx_seqs = [] # 存储符合条件的索引
        if use_sequence and self.seq_len > 1:
            for idx in idxs:
                # 剔除不满足序列长度的样本
                need = self.seq_len # 需要的历史长度
                if idx + 1 < need:
                    continue
                start = idx - need + 1
                # 剔除跨回合边界的样本
                window_dones = dones_arr[start : idx + 1] # 采样窗口内的 done 标志
                if window_dones[:-1].sum() != 0: # 中间有 done 标志，说明跨回合
                    continue
                idx_seqs.append((start, idx + 1)) # 记录有效的起止索引
        
        # 从有效索引中随机选择所需数量并返回
        if not use_sequence or self.seq_len == 1:
            idxs = np.random.choice(idxs, size=batch_size if len(idxs) >= batch_size else len(idxs), replace=False) # 重新不放回采样
            return idxs # 返回采样索引，形状为 (batch_size,)
        elif use_sequence and self.seq_len > 1:
            idx_seqs = np.random.choice(len(idx_seqs), size=batch_size if len(idx_seqs) >= batch_size else len(idx_seqs), replace=False) # 重新不放回采样
            return np.array(idx_seqs) # 返回采样结束索引，形状为 (batch_size, 2)
        else:
            idxs = np.array([]) # 空数组
            raise ValueError("取样没有得到有效索引")


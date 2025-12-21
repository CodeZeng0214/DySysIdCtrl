import numpy as np
import torch
from typing import Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Replay buffer supporting MLP and GRU (sequence) sampling."""

    def __init__(self, state_dim: int, capacity: int = 100000, seq_len: int = 1) -> None:
        self.state_dim = state_dim
        self.capacity = capacity
        self.seq_len = max(1, seq_len)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.delay_steps = np.zeros((capacity, 1), dtype=np.int32)
        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool, delay: int = 0) -> None:
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.delay_steps[idx] = int(delay)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, use_sequence: bool = False) -> Tuple[torch.Tensor, ...]:
        assert self.size >= batch_size, "Not enough samples"
        if not use_sequence or self.seq_len == 1:
            idx = np.random.randint(0, self.size, size=batch_size)
            states = torch.tensor(self.states[idx], device=device)
            actions = torch.tensor(self.actions[idx], device=device)
            rewards = torch.tensor(self.rewards[idx], device=device)
            next_states = torch.tensor(self.next_states[idx], device=device)
            dones = torch.tensor(self.dones[idx], device=device)
            delays = torch.tensor(self.delay_steps[idx], device=device)
            return states, actions, rewards, next_states, dones, delays

        # sequence sampling requires contiguous windows within an episode (no done inside the window except maybe last)
        valid_starts = []
        upper = self.size - self.seq_len
        base = (self.ptr - self.size) % self.capacity  # chronological start index in circular buffer
        for s in range(0, max(0, upper) + 1):
            idx_window = (base + np.arange(s, s + self.seq_len)) % self.capacity
            window_dones = self.dones[idx_window]
            if window_dones[:-1].sum() == 0:  # allow done at last position only
                valid_starts.append(s)
        if not valid_starts:
            raise ValueError("No valid sequence windows available; try shorter seq_len or more data")

        starts = np.random.choice(valid_starts, size=batch_size)
        idx_seq = np.stack([(base + np.arange(s, s + self.seq_len)) % self.capacity for s in starts], axis=0)

        states = torch.tensor(self.states[idx_seq], device=device)
        actions = torch.tensor(self.actions[idx_seq[:, -1]], device=device)  # last action for TD target
        rewards = torch.tensor(self.rewards[idx_seq[:, -1]], device=device)
        next_states = torch.tensor(self.next_states[idx_seq], device=device)
        dones = torch.tensor(self.dones[idx_seq[:, -1]], device=device)
        delays = torch.tensor(self.delay_steps[idx_seq[:, -1]], device=device)
        return states, actions, rewards, next_states, dones, delays

    def reset(self) -> None:
        self.ptr = 0
        self.size = 0

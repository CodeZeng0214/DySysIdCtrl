import torch
import torch.nn as nn
from typing import Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPActor(nn.Module):
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


class MLPCritic(nn.Module):
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


class GRUActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_bound: float, gru_hidden: int, gru_layers: int, use_attention: bool = True) -> None:
        super().__init__()
        self.encoder = GRUEncoder(state_dim, gru_hidden, gru_layers, use_attention)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.action_bound = action_bound
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        context = self.encoder(state_seq)
        return self.head(context) * self.action_bound


class GRUCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, gru_hidden: int, gru_layers: int, use_attention: bool = True) -> None:
        super().__init__()
        self.encoder = GRUEncoder(state_dim, gru_hidden, gru_layers, use_attention)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        context = self.encoder(state_seq)
        return self.head(torch.cat([context, action], dim=-1))


def build_actor_critic(
    arch: str,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    action_bound: float,
    gru_hidden: int = 64,
    gru_layers: int = 1,
    use_attention: bool = True,
) -> Tuple[nn.Module, nn.Module]:
    arch = arch.lower()
    if arch == "mlp":
        actor = MLPActor(state_dim, action_dim, hidden_dim, action_bound)
        critic = MLPCritic(state_dim, action_dim, hidden_dim)
    elif arch == "gru":
        actor = GRUActor(state_dim, action_dim, hidden_dim, action_bound, gru_hidden, gru_layers, use_attention)
        critic = GRUCritic(state_dim, action_dim, hidden_dim, gru_hidden, gru_layers, use_attention)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return actor.to(device), critic.to(device)

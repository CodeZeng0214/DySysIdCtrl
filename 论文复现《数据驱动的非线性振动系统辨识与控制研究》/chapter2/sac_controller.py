"""
SAC (Soft Actor-Critic) Controller for Van der Pol System
Based on Chapter 2 of "Data-driven Identification and Control of Nonlinear Vibration Systems"

This module implements SAC reinforcement learning for optimal control of the Van der Pol
oscillator. SAC is an off-policy actor-critic algorithm that maximizes both expected
return and entropy for better exploration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from van_der_pol_system import VanDerPolSystem


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: float, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.FloatTensor([t[1] for t in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([t[2] for t in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.BoolTensor([t[4] for t in batch]).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for SAC
    Outputs mean and log standard deviation for a Gaussian distribution
    """
    
    def __init__(self, state_dim: int = 2, action_dim: int = 1, 
                 hidden_dim: int = 256, max_action: float = 5.0):
        super(GaussianPolicy, self).__init__()
        
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
            
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy
        
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(x_t) * self.max_action
        
        # Calculate log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(x_t).pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for evaluation (deterministic or stochastic)"""
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean) * self.max_action
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.sample()
            action = torch.tanh(x_t) * self.max_action
            
        return action


class QNetwork(nn.Module):
    """Q-value network for SAC"""
    
    def __init__(self, state_dim: int = 2, action_dim: int = 1, 
                 hidden_dim: int = 256):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
            
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SACController:
    """
    SAC Controller for Van der Pol System
    
    Implementation of Soft Actor-Critic for optimal control of nonlinear
    vibration systems with automatic entropy tuning.
    """
    
    def __init__(self,
                 mu: float = 1.0,
                 state_dim: int = 2,
                 action_dim: int = 1,
                 max_action: float = 5.0,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True):
        """
        Initialize SAC controller
        
        Args:
            mu: Van der Pol parameter
            state_dim: State dimension
            action_dim: Action dimension  
            max_action: Maximum action value
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Temperature parameter for entropy regularization
            automatic_entropy_tuning: Whether to automatically tune alpha
        """
        self.mu = mu
        self.system = VanDerPolSystem(mu=mu)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy = GaussianPolicy(state_dim, action_dim, max_action=max_action).to(self.device)
        
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        
        self.q1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim).to(self.device)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # Entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha).to(self.device)
            
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training statistics
        self.policy_losses = []
        self.q1_losses = []
        self.q2_losses = []
        self.alpha_losses = []
        self.alpha_values = []
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """
        Select action using the policy network
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy.get_action(state_tensor, deterministic).cpu().data.numpy().flatten()[0]
        return action
    
    def compute_reward(self, state: np.ndarray, action: float, next_state: np.ndarray) -> float:
        """
        Compute reward for the given transition
        
        Reward function designed to stabilize the system:
        - Quadratic state cost
        - Control effort penalty
        - Stability bonus
        - Progress reward
        
        Args:
            state: Current state
            action: Applied action
            next_state: Next state
            
        Returns:
            Reward value
        """
        # State cost (quadratic penalty for deviation from origin)
        Q_matrix = np.array([[10.0, 0.0], [0.0, 1.0]])
        state_cost = next_state.T @ Q_matrix @ next_state
        
        # Control effort penalty
        control_cost = 0.1 * action**2
        
        # Distance from origin
        current_distance = np.sqrt(np.sum(state**2))
        next_distance = np.sqrt(np.sum(next_state**2))
        
        # Progress reward (getting closer to origin)
        progress_reward = current_distance - next_distance
        
        # Stability bonus
        stability_bonus = 0.0
        if next_distance < 0.1:
            stability_bonus = 2.0
        elif next_distance < 0.5:
            stability_bonus = 1.0
        
        # Penalty for large states (encourage staying in reasonable region)
        boundary_penalty = 0.0
        if next_distance > 5.0:
            boundary_penalty = -10.0
        
        # Combined reward
        reward = -state_cost - control_cost + progress_reward + stability_bonus + boundary_penalty
        
        return float(reward)
    
    def train_step(self, batch_size: int = 256) -> Tuple[float, float, float, float]:
        """
        Perform one training step
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Policy loss, Q1 loss, Q2 loss, Alpha loss
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0, 0.0
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (~dones) * self.gamma * target_q
        
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha (temperature parameter)
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()
        
        # Soft update target networks
        self._soft_update(self.q1_target, self.q1)
        self._soft_update(self.q2_target, self.q2)
        
        return policy_loss.item(), q1_loss.item(), q2_loss.item(), alpha_loss
    
    def _soft_update(self, target_net: nn.Module, source_net: nn.Module):
        """Soft update target network parameters"""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_episodes(self,
                      num_episodes: int = 1000,
                      max_steps: int = 200,
                      eval_freq: int = 100) -> dict:
        """
        Train the SAC controller
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            eval_freq: Frequency of evaluation episodes
            
        Returns:
            Training statistics
        """
        episode_rewards = []
        eval_rewards = []
        eval_episodes = []
        
        print(f"Training SAC Controller on Van der Pol System (μ={self.mu})")
        print("=" * 60)
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.system.reset()
            episode_reward = 0.0
            
            for step in range(max_steps):
                # Select action (stochastic during training)
                action = self.select_action(state, deterministic=False)
                
                # Take step in environment
                next_state = self.system.step(action)
                reward = self.compute_reward(state, action, next_state)
                done = self.system.is_done()
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.replay_buffer) > 1000:
                    policy_loss, q1_loss, q2_loss, alpha_loss = self.train_step()
                    self.policy_losses.append(policy_loss)
                    self.q1_losses.append(q1_loss)
                    self.q2_losses.append(q2_loss)
                    self.alpha_losses.append(alpha_loss)
                    self.alpha_values.append(self.alpha.item())
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Evaluation
            if episode % eval_freq == 0:
                eval_reward = self._evaluate()
                eval_rewards.append(eval_reward)
                eval_episodes.append(episode)
                
                alpha_str = f"{self.alpha.item():.4f}" if self.automatic_entropy_tuning else f"{self.alpha:.4f}"
                print(f"Episode {episode:4d} | "
                      f"Train Reward: {episode_reward:7.2f} | "
                      f"Eval Reward: {eval_reward:7.2f} | "
                      f"Alpha: {alpha_str} | "
                      f"Buffer Size: {len(self.replay_buffer):6d}")
        
        return {
            'episode_rewards': episode_rewards,
            'eval_rewards': eval_rewards,
            'eval_episodes': eval_episodes,
            'policy_losses': self.policy_losses,
            'q1_losses': self.q1_losses,
            'q2_losses': self.q2_losses,
            'alpha_losses': self.alpha_losses,
            'alpha_values': self.alpha_values
        }
    
    def _evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the current policy deterministically"""
        total_reward = 0.0
        
        for _ in range(num_episodes):
            state = self.system.reset()
            episode_reward = 0.0
            
            for _ in range(200):
                action = self.select_action(state, deterministic=True)
                next_state = self.system.step(action)
                reward = self.compute_reward(state, action, next_state)
                
                state = next_state
                episode_reward += reward
                
                if self.system.is_done():
                    break
                    
            total_reward += episode_reward
            
        return total_reward / num_episodes
    
    def simulate_control(self,
                        initial_state: np.ndarray,
                        t_span: Tuple[float, float] = (0, 10),
                        dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the trained controller
        
        Args:
            initial_state: Initial state
            t_span: Time span
            dt: Time step
            
        Returns:
            Time array, state trajectory, control trajectory
        """
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)
        n_steps = len(t)
        
        states = np.zeros((n_steps, 2))
        controls = np.zeros(n_steps)
        
        # Set initial state
        self.system.state = initial_state.copy()
        states[0] = initial_state
        
        # Simulate
        for i in range(n_steps - 1):
            action = self.select_action(states[i], deterministic=True)
            controls[i] = action
            
            next_state = self.system.rk4_step(states[i], action, dt)
            states[i + 1] = next_state
            
        controls[-1] = self.select_action(states[-1], deterministic=True)
        
        return t, states, controls
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'mu': self.mu,
            'max_action': self.max_action
        }, path)
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp()


def main():
    """Demonstration of SAC controller training"""
    print("SAC Controller for Van der Pol System")
    print("=" * 50)
    
    # Create controller
    controller = SACController(mu=1.0, max_action=5.0, automatic_entropy_tuning=True)
    
    # Train controller
    training_stats = controller.train_episodes(
        num_episodes=500,
        max_steps=200,
        eval_freq=50
    )
    
    # Test trained controller
    initial_states = [
        np.array([2.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([1.5, -1.5])
    ]
    
    for i, x0 in enumerate(initial_states):
        print(f"\nTesting with initial condition: {x0}")
        
        t, states, controls = controller.simulate_control(
            initial_state=x0,
            t_span=(0, 10),
            dt=0.01
        )
        
        # Calculate performance metrics
        final_distance = np.sqrt(np.sum(states[-1]**2))
        control_effort = np.sum(controls**2) * 0.01
        
        print(f"Final distance from origin: {final_distance:.4f}")
        print(f"Control effort: {control_effort:.2f}")


if __name__ == "__main__":
    main()

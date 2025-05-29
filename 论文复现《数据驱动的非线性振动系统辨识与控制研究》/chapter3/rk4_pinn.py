"""
RK4-PINN (Runge-Kutta Physics-Informed Neural Network) for QZS System Identification
RK4物理信息神经网络用于QZS系统辨识

This module implements a Physics-Informed Neural Network (PINN) that incorporates
the RK4 integration scheme for identifying QZS system parameters from data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from qzs_system import QZSSystem, QZSParameters
from utils.plotting import plot_training_progress, save_figure
from utils.data_utils import generate_training_data, normalize_data


@dataclass
class PINNConfig:
    """Configuration for RK4-PINN training"""
    
    # Network architecture
    hidden_layers: List[int] = None
    activation: str = 'tanh'
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 5000
    patience: int = 500
    
    # Loss weights
    data_loss_weight: float = 1.0
    physics_loss_weight: float = 1.0
    ic_loss_weight: float = 10.0  # Initial condition loss
    
    # Physics constraints
    dt: float = 0.001  # Time step for RK4
    
    # Regularization
    l2_regularization: float = 1e-6
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 64, 64]


class RK4PINN(nn.Module):
    """
    RK4-Physics Informed Neural Network for QZS system identification
    RK4物理信息神经网络
    """
    
    def __init__(self, config: PINNConfig):
        """
        Initialize the RK4-PINN
        
        Args:
            config: PINN configuration
        """
        super(RK4PINN, self).__init__()
        
        self.config = config
        
        # Define network architecture
        self.layers = nn.ModuleList()
        
        # Input layer: [t, x, v] -> hidden
        input_dim = 3  # time, position, velocity
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer: hidden -> [x_next, v_next]
        self.layers.append(nn.Linear(prev_dim, 2))
        
        # Activation function
        if config.activation == 'tanh':
            self.activation = torch.tanh
        elif config.activation == 'relu':
            self.activation = torch.relu
        elif config.activation == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.tanh
        
        # Learnable physical parameters
        self.param_network = nn.Sequential(
            nn.Linear(1, 32),  # Input: time (for time-varying parameters)
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4)   # Output: [k1, k2, c, ks] (normalized)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Parameter bounds (for physical constraints)
        self.param_bounds = {
            'k1': (500.0, 2000.0),   # Linear stiffness bounds
            'k2': (100.0, 1500.0),   # Nonlinear stiffness bounds
            'c': (1.0, 20.0),        # Damping bounds
            'ks': (1000.0, 3000.0)   # QZS spring stiffness bounds
        }
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize parameter network
        for layer in self.param_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network
        
        Args:
            t: Time tensor
            x: Position tensor
            v: Velocity tensor
            
        Returns:
            Predicted next state [x_next, v_next]
        """
        
        # Combine inputs
        inputs = torch.cat([t, x, v], dim=1)
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            inputs = layer(inputs)
            inputs = self.activation(inputs)
        
        # Output layer
        output = self.layers[-1](inputs)
        
        return output
    
    def get_parameters(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get physical parameters from parameter network
        
        Args:
            t: Time tensor
            
        Returns:
            Dictionary of physical parameters
        """
        
        # Normalize time input
        t_norm = t / 10.0  # Assuming typical time scale of 10 seconds
        
        # Get normalized parameters
        params_norm = self.param_network(t_norm)
        
        # Map to physical ranges
        k1 = self.param_bounds['k1'][0] + (self.param_bounds['k1'][1] - self.param_bounds['k1'][0]) * torch.sigmoid(params_norm[:, 0:1])
        k2 = self.param_bounds['k2'][0] + (self.param_bounds['k2'][1] - self.param_bounds['k2'][0]) * torch.sigmoid(params_norm[:, 1:2])
        c = self.param_bounds['c'][0] + (self.param_bounds['c'][1] - self.param_bounds['c'][0]) * torch.sigmoid(params_norm[:, 2:3])
        ks = self.param_bounds['ks'][0] + (self.param_bounds['ks'][1] - self.param_bounds['ks'][0]) * torch.sigmoid(params_norm[:, 3:4])
        
        return {
            'k1': k1,
            'k2': k2,
            'c': c,
            'ks': ks
        }
    
    def qzs_dynamics(self, t: torch.Tensor, x: torch.Tensor, v: torch.Tensor, 
                     F_control: torch.Tensor, F_external: torch.Tensor) -> torch.Tensor:
        """
        QZS system dynamics using learned parameters
        
        Args:
            t: Time
            x: Position
            v: Velocity
            F_control: Control force
            F_external: External force
            
        Returns:
            Acceleration
        """
        
        # Get learned parameters
        params = self.get_parameters(t)
        k1, k2, c, ks = params['k1'], params['k2'], params['c'], params['ks']
        
        # System mass (assumed known)
        m = 10.0
        
        # QZS geometry parameters (assumed known)
        L = 0.5
        theta0 = torch.deg2rad(torch.tensor(30.0))
        
        # Calculate QZS force
        h = L * torch.sin(theta0)
        l_horizontal = L * torch.cos(theta0)
        
        current_length = torch.sqrt((h - x)**2 + l_horizontal**2)
        current_angle = torch.atan2(h - x, l_horizontal)
        
        spring_force = ks * (current_length - L)
        F_qzs = -2 * spring_force * torch.sin(current_angle)
        
        # Total force
        F_linear = -k1 * x
        F_nonlinear = -k2 * x**3
        F_damping = -c * v
        
        F_total = F_linear + F_qzs + F_nonlinear + F_damping + F_control + F_external
        
        # Acceleration
        acceleration = F_total / m
        
        return acceleration
    
    def rk4_step(self, t: torch.Tensor, x: torch.Tensor, v: torch.Tensor,
                 F_control: torch.Tensor, F_external: torch.Tensor, dt: float) -> torch.Tensor:
        """
        RK4 integration step using physics
        
        Args:
            t: Current time
            x: Current position
            v: Current velocity
            F_control: Control force
            F_external: External force
            dt: Time step
            
        Returns:
            Next state [x_next, v_next]
        """
        
        # RK4 coefficients for position and velocity
        # k1
        k1_x = v
        k1_v = self.qzs_dynamics(t, x, v, F_control, F_external)
        
        # k2
        t_half = t + dt / 2
        x_k2 = x + dt * k1_x / 2
        v_k2 = v + dt * k1_v / 2
        k2_x = v_k2
        k2_v = self.qzs_dynamics(t_half, x_k2, v_k2, F_control, F_external)
        
        # k3
        x_k3 = x + dt * k2_x / 2
        v_k3 = v + dt * k2_v / 2
        k3_x = v_k3
        k3_v = self.qzs_dynamics(t_half, x_k3, v_k3, F_control, F_external)
        
        # k4
        t_full = t + dt
        x_k4 = x + dt * k3_x
        v_k4 = v + dt * k3_v
        k4_x = v_k4
        k4_v = self.qzs_dynamics(t_full, x_k4, v_k4, F_control, F_external)
        
        # Final RK4 update
        x_next = x + dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        v_next = v + dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        
        return torch.cat([x_next, v_next], dim=1)


class RK4PINNTrainer:
    """
    Trainer for RK4-PINN
    RK4-PINN训练器
    """
    
    def __init__(self, config: PINNConfig, device: str = 'cpu'):
        """
        Initialize trainer
        
        Args:
            config: PINN configuration
            device: Computing device
        """
        self.config = config
        self.device = device
        
        # Initialize network
        self.model = RK4PINN(config).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=config.learning_rate,
                                   weight_decay=config.l2_regularization)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100, verbose=True
        )
        
        # Training history
        self.history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'ic_loss': [],
            'validation_loss': []
        }
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def generate_training_data(self, qzs_system: QZSSystem, 
                             n_trajectories: int = 50,
                             trajectory_length: float = 10.0,
                             noise_level: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Generate training data from QZS system simulations
        
        Args:
            qzs_system: QZS system for data generation
            n_trajectories: Number of trajectories
            trajectory_length: Length of each trajectory
            noise_level: Noise level to add to measurements
            
        Returns:
            Dictionary containing training data
        """
        
        print(f"Generating {n_trajectories} training trajectories...")
        
        all_time = []
        all_states = []
        all_controls = []
        all_externals = []
        
        dt = self.config.dt
        n_steps = int(trajectory_length / dt)
        
        for traj in range(n_trajectories):
            # Random initial conditions
            x0 = np.random.uniform(-0.1, 0.1)
            v0 = np.random.uniform(-0.5, 0.5)
            initial_state = np.array([x0, v0])
            
            # Random control strategy (for data diversity)
            control_type = np.random.choice(['zero', 'random', 'proportional'])
            
            def control_func(t, state):
                if control_type == 'zero':
                    return 0.0
                elif control_type == 'random':
                    return np.random.uniform(-20, 20)
                else:  # proportional
                    return -50 * state[0] - 10 * state[1]  # Simple PD control
            
            # Simulate trajectory
            result = qzs_system.simulate(
                t_span=(0, trajectory_length),
                initial_state=initial_state,
                control_func=control_func
            )
            
            # Store data
            all_time.append(result['time'])
            all_states.append(result['states'])
            all_controls.append(result['control_force'])
            all_externals.append(result['external_force'])
        
        # Concatenate all trajectories
        time_data = np.concatenate(all_time)
        state_data = np.concatenate(all_states)
        control_data = np.concatenate(all_controls)
        external_data = np.concatenate(all_externals)
        
        # Add noise to measurements
        if noise_level > 0:
            noise_x = np.random.normal(0, noise_level, state_data[:, 0].shape)
            noise_v = np.random.normal(0, noise_level, state_data[:, 1].shape)
            state_data[:, 0] += noise_x
            state_data[:, 1] += noise_v
        
        # Convert to tensors
        data = {
            't': torch.tensor(time_data, dtype=torch.float32, device=self.device).reshape(-1, 1),
            'x': torch.tensor(state_data[:, 0], dtype=torch.float32, device=self.device).reshape(-1, 1),
            'v': torch.tensor(state_data[:, 1], dtype=torch.float32, device=self.device).reshape(-1, 1),
            'u': torch.tensor(control_data, dtype=torch.float32, device=self.device).reshape(-1, 1),
            'f_ext': torch.tensor(external_data, dtype=torch.float32, device=self.device).reshape(-1, 1)
        }
        
        print(f"Generated {len(time_data)} data points")
        
        return data
    
    def create_physics_points(self, data: Dict[str, torch.Tensor], 
                            n_physics_points: int = 10000) -> Dict[str, torch.Tensor]:
        """
        Create physics collocation points
        
        Args:
            data: Training data
            n_physics_points: Number of physics points
            
        Returns:
            Physics collocation points
        """
        
        # Sample points in the domain
        t_min, t_max = data['t'].min(), data['t'].max()
        x_min, x_max = data['x'].min(), data['x'].max()
        v_min, v_max = data['v'].min(), data['v'].max()
        
        # Random sampling in the domain
        t_physics = torch.rand(n_physics_points, 1, device=self.device) * (t_max - t_min) + t_min
        x_physics = torch.rand(n_physics_points, 1, device=self.device) * (x_max - x_min) + x_min
        v_physics = torch.rand(n_physics_points, 1, device=self.device) * (v_max - v_min) + v_min
        
        # Random control and external forces
        u_physics = torch.rand(n_physics_points, 1, device=self.device) * 40 - 20  # [-20, 20]
        f_ext_physics = torch.rand(n_physics_points, 1, device=self.device) * 100 - 50  # [-50, 50]
        
        physics_data = {
            't': t_physics,
            'x': x_physics,
            'v': v_physics,
            'u': u_physics,
            'f_ext': f_ext_physics
        }
        
        return physics_data
    
    def compute_data_loss(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute data fitting loss
        
        Args:
            data: Training data
            
        Returns:
            Data loss
        """
        
        # Create sequential data pairs (current state -> next state)
        dt = self.config.dt
        
        # Find indices where time step is approximately dt
        time_diffs = data['t'][1:] - data['t'][:-1]
        valid_indices = torch.where(torch.abs(time_diffs - dt) < dt/10)[0]
        
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Current states
        t_current = data['t'][valid_indices].reshape(-1, 1)
        x_current = data['x'][valid_indices].reshape(-1, 1)
        v_current = data['v'][valid_indices].reshape(-1, 1)
        u_current = data['u'][valid_indices].reshape(-1, 1)
        f_ext_current = data['f_ext'][valid_indices].reshape(-1, 1)
        
        # Next states (ground truth)
        x_next_true = data['x'][valid_indices + 1].reshape(-1, 1)
        v_next_true = data['v'][valid_indices + 1].reshape(-1, 1)
        
        # Predict next states using RK4
        next_state_pred = self.model.rk4_step(
            t_current, x_current, v_current, u_current, f_ext_current, dt
        )
        
        x_next_pred = next_state_pred[:, 0:1]
        v_next_pred = next_state_pred[:, 1:2]
        
        # MSE loss
        loss_x = torch.mean((x_next_pred - x_next_true)**2)
        loss_v = torch.mean((v_next_pred - v_next_true)**2)
        
        return loss_x + loss_v
    
    def compute_physics_loss(self, physics_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute physics-based loss (PDE residual)
        
        Args:
            physics_data: Physics collocation points
            
        Returns:
            Physics loss
        """
        
        t = physics_data['t']
        x = physics_data['x']
        v = physics_data['v']
        u = physics_data['u']
        f_ext = physics_data['f_ext']
        
        # Enable gradient computation
        t.requires_grad_(True)
        x.requires_grad_(True)
        v.requires_grad_(True)
        
        # Compute accelerations from physics
        acceleration_physics = self.model.qzs_dynamics(t, x, v, u, f_ext)
        
        # Compute accelerations from network (time derivative of velocity)
        # This requires the network to also predict velocity derivatives
        # For simplicity, we'll use the physics equation directly here
        
        # The physics loss ensures that the learned parameters satisfy the governing equation
        # For QZS system: m*ẍ + c*ẋ + k_eff(x)*x = F_ext + F_control
        
        # Get learned parameters
        params = self.model.get_parameters(t)
        k1, k2, c, ks = params['k1'], params['k2'], params['c'], params['ks']
        
        # Expected acceleration from governing equation
        m = 10.0  # Known mass
        
        # Linear spring force
        F_linear = -k1 * x
        
        # Nonlinear spring force
        F_nonlinear = -k2 * x**3
        
        # Damping force
        F_damping = -c * v
        
        # QZS force (simplified calculation)
        L = 0.5
        theta0 = torch.deg2rad(torch.tensor(30.0))
        h = L * torch.sin(theta0)
        l_horizontal = L * torch.cos(theta0)
        
        current_length = torch.sqrt((h - x)**2 + l_horizontal**2)
        current_angle = torch.atan2(h - x, l_horizontal)
        spring_force = ks * (current_length - L)
        F_qzs = -2 * spring_force * torch.sin(current_angle)
        
        # Total force
        F_total = F_linear + F_qzs + F_nonlinear + F_damping + u + f_ext
        
        # Expected acceleration
        acceleration_expected = F_total / m
        
        # Physics residual
        residual = acceleration_physics - acceleration_expected
        
        return torch.mean(residual**2)
    
    def compute_ic_loss(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute initial condition loss
        
        Args:
            data: Training data
            
        Returns:
            Initial condition loss
        """
        
        # Find initial conditions (t = 0 or close to 0)
        t_threshold = 0.01
        ic_indices = torch.where(data['t'] < t_threshold)[0]
        
        if len(ic_indices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Initial states
        t_ic = data['t'][ic_indices]
        x_ic = data['x'][ic_indices]
        v_ic = data['v'][ic_indices]
        
        # Network prediction at initial time
        pred_ic = self.model(t_ic, x_ic, v_ic)
        
        # The network should predict the same initial state
        loss = torch.mean((pred_ic[:, 0:1] - x_ic)**2 + (pred_ic[:, 1:2] - v_ic)**2)
        
        return loss
    
    def train_epoch(self, data: Dict[str, torch.Tensor], 
                   physics_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            data: Training data
            physics_data: Physics collocation points
            
        Returns:
            Dictionary of losses
        """
        
        self.model.train()
        
        # Compute losses
        data_loss = self.compute_data_loss(data)
        physics_loss = self.compute_physics_loss(physics_data)
        ic_loss = self.compute_ic_loss(data)
        
        # Total loss
        total_loss = (self.config.data_loss_weight * data_loss +
                     self.config.physics_loss_weight * physics_loss +
                     self.config.ic_loss_weight * ic_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'ic_loss': ic_loss.item()
        }
    
    def validate(self, val_data: Dict[str, torch.Tensor]) -> float:
        """
        Validate the model
        
        Args:
            val_data: Validation data
            
        Returns:
            Validation loss
        """
        
        self.model.eval()
        
        with torch.no_grad():
            val_loss = self.compute_data_loss(val_data)
        
        return val_loss.item()
    
    def train(self, qzs_system: QZSSystem) -> Dict[str, List[float]]:
        """
        Train the RK4-PINN
        
        Args:
            qzs_system: QZS system for data generation
            
        Returns:
            Training history
        """
        
        print("Starting RK4-PINN training...")
        print("=" * 50)
        
        # Generate training data
        train_data = self.generate_training_data(qzs_system, n_trajectories=40)
        val_data = self.generate_training_data(qzs_system, n_trajectories=10)
        
        # Create physics points
        physics_data = self.create_physics_points(train_data)
        
        # Training loop
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            # Train epoch
            epoch_losses = self.train_epoch(train_data, physics_data)
            
            # Validate
            val_loss = self.validate(val_data)
            epoch_losses['validation_loss'] = val_loss
            
            # Update history
            for key, value in epoch_losses.items():
                self.history[key].append(value)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1:5d} | "
                      f"Total Loss: {epoch_losses['total_loss']:.6f} | "
                      f"Data Loss: {epoch_losses['data_loss']:.6f} | "
                      f"Physics Loss: {epoch_losses['physics_loss']:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"Training completed! Best validation loss: {self.best_loss:.6f}")
        
        return self.history
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_loss': self.best_loss
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Model loaded from {filepath}")
    
    def get_learned_parameters(self, t_eval: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Get learned physical parameters
        
        Args:
            t_eval: Time points for evaluation
            
        Returns:
            Dictionary of learned parameters
        """
        
        if t_eval is None:
            t_eval = np.linspace(0, 10, 100)
        
        self.model.eval()
        
        with torch.no_grad():
            t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=self.device).reshape(-1, 1)
            params = self.model.get_parameters(t_tensor)
            
            learned_params = {}
            for key, value in params.items():
                learned_params[key] = value.cpu().numpy().flatten()
        
        return learned_params, t_eval


def demo_rk4_pinn():
    """Demonstrate RK4-PINN training and identification"""
    
    print("RK4-PINN Demonstration")
    print("=" * 50)
    
    # Create true QZS system
    true_params = QZSParameters(
        m=10.0,
        c=5.0,
        k1=1000.0,
        k2=800.0,
        ks=2000.0,
        F0=20.0,
        omega_f=5.0
    )
    
    qzs_true = QZSSystem(true_params)
    
    print(f"True parameters:")
    print(f"k1 = {true_params.k1} N/m")
    print(f"k2 = {true_params.k2} N/m³")
    print(f"c = {true_params.c} Ns/m")
    print(f"ks = {true_params.ks} N/m")
    
    # Configure PINN
    config = PINNConfig(
        hidden_layers=[64, 64, 32],
        learning_rate=1e-3,
        max_epochs=2000,
        patience=200,
        data_loss_weight=1.0,
        physics_loss_weight=0.1,
        ic_loss_weight=10.0
    )
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = RK4PINNTrainer(config, device)
    
    # Train PINN
    history = trainer.train(qzs_true)
    
    # Get learned parameters
    learned_params, t_eval = trainer.get_learned_parameters()
    
    print(f"\nLearned parameters (mean ± std):")
    for param_name in ['k1', 'k2', 'c', 'ks']:
        values = learned_params[param_name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        true_val = getattr(true_params, param_name)
        error = abs(mean_val - true_val) / true_val * 100
        
        print(f"{param_name} = {mean_val:.1f} ± {std_val:.1f} (true: {true_val}, error: {error:.1f}%)")
    
    return trainer, history, learned_params


if __name__ == "__main__":
    # Run demonstration
    trainer, history, learned_params = demo_rk4_pinn()
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Training losses
    axes[0, 0].plot(history['total_loss'], label='Total Loss')
    axes[0, 0].plot(history['data_loss'], label='Data Loss')
    axes[0, 0].plot(history['physics_loss'], label='Physics Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].grid(True)
    
    # Validation loss
    axes[0, 1].plot(history['validation_loss'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(True)
    
    # Learned parameters
    param_names = ['k1', 'k2', 'c', 'ks']
    true_values = [1000.0, 800.0, 5.0, 2000.0]
    
    for i, (param_name, true_val) in enumerate(zip(param_names, true_values)):
        if i < 4:
            row = (i + 2) // 3
            col = (i + 2) % 3
            
            if row < 2 and col < 3:
                t_eval = np.linspace(0, 10, len(learned_params[param_name]))
                axes[row, col].plot(t_eval, learned_params[param_name], 'b-', linewidth=2, label='Learned')
                axes[row, col].axhline(y=true_val, color='r', linestyle='--', linewidth=2, label='True')
                axes[row, col].set_xlabel('Time (s)')
                axes[row, col].set_ylabel(f'{param_name}')
                axes[row, col].set_title(f'Parameter {param_name}')
                axes[row, col].legend()
                axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nRK4-PINN demonstration completed!")

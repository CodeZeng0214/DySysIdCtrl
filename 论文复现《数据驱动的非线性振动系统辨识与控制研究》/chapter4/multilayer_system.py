"""
Chapter 4: Multi-layer Nonlinear Vibration Isolation System
Implements a multi-layer vibration isolation system with nonlinear characteristics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from utils.math_utils import runge_kutta_4


class MultiLayerSystem:
    """Multi-layer nonlinear vibration isolation system"""
    
    def __init__(self, n_layers=3):
        """
        Initialize multi-layer system
        
        Args:
            n_layers: Number of isolation layers
        """
        self.n_layers = n_layers
        self.dof = n_layers + 1  # DOF = layers + base
        
        # System parameters for each layer
        self.masses = np.array([50.0, 20.0, 10.0, 5.0])[:self.dof]  # kg
        self.linear_stiffness = np.array([50000.0, 30000.0, 20000.0, 15000.0])[:self.dof]  # N/m
        self.nonlinear_stiffness = np.array([1e6, 8e5, 6e5, 4e5])[:self.dof]  # N/m^3
        self.damping = np.array([100.0, 80.0, 60.0, 40.0])[:self.dof]  # N·s/m
        
        # Geometric parameters for QZS elements
        self.h0 = 0.02  # Initial height (m)
        self.a = 0.05   # Horizontal distance (m)
        
        # Control parameters
        self.control_limit = 1000.0  # Maximum control force (N)
        
        # State dimension: [x1, x2, ..., xn, v1, v2, ..., vn]
        self.state_dim = 2 * self.dof
        
    def qzs_force(self, x, layer_idx):
        """
        Calculate quasi-zero-stiffness force for a layer
        
        Args:
            x: Relative displacement
            layer_idx: Layer index
            
        Returns:
            QZS force
        """
        h = self.h0 + x
        if h <= 0:
            h = 1e-6  # Avoid division by zero
            
        # Geometric relationships
        l = np.sqrt(self.a**2 + h**2)
        l0 = np.sqrt(self.a**2 + self.h0**2)
        
        # QZS force components
        vertical_force = 2 * self.linear_stiffness[layer_idx] * (l - l0) * h / l
        nonlinear_force = self.nonlinear_stiffness[layer_idx] * x**3
        
        return vertical_force + nonlinear_force
    
    def system_dynamics(self, t, state, control_forces=None, base_excitation=None):
        """
        System dynamics: dx/dt = f(x, u, w)
        
        Args:
            t: Time
            state: System state [x1, x2, ..., xn, v1, v2, ..., vn]
            control_forces: Control forces for each layer
            base_excitation: Base excitation function
            
        Returns:
            State derivative
        """
        n = self.dof
        positions = state[:n]
        velocities = state[n:]
        
        if control_forces is None:
            control_forces = np.zeros(n)
        
        if base_excitation is None:
            base_disp = 0.0
            base_vel = 0.0
        else:
            base_disp = base_excitation(t)
            # Numerical derivative for base velocity
            dt = 1e-6
            base_vel = (base_excitation(t + dt) - base_excitation(t - dt)) / (2 * dt)
        
        # Clip control forces
        control_forces = np.clip(control_forces, -self.control_limit, self.control_limit)
        
        # Calculate forces
        forces = np.zeros(n)
        
        for i in range(n):
            # Relative displacement and velocity
            if i == 0:  # First layer connected to base
                rel_disp = positions[i] - base_disp
                rel_vel = velocities[i] - base_vel
            else:  # Connected to previous layer
                rel_disp = positions[i] - positions[i-1]
                rel_vel = velocities[i] - velocities[i-1]
            
            # QZS force (downward connection)
            qzs_force_down = self.qzs_force(rel_disp, i)
            
            # Damping force (downward connection)
            damping_force_down = self.damping[i] * rel_vel
            
            # Total force from lower connection
            force_down = qzs_force_down + damping_force_down
            
            # Force from upper connection (if exists)
            force_up = 0.0
            if i < n - 1:
                rel_disp_up = positions[i+1] - positions[i]
                rel_vel_up = velocities[i+1] - velocities[i]
                
                qzs_force_up = self.qzs_force(rel_disp_up, i+1)
                damping_force_up = self.damping[i+1] * rel_vel_up
                force_up = qzs_force_up + damping_force_up
            
            # Equation of motion: m * a = F_up - F_down + F_control
            forces[i] = (force_up - force_down + control_forces[i]) / self.masses[i]
        
        # State derivative
        state_dot = np.zeros(2 * n)
        state_dot[:n] = velocities  # dx/dt = v
        state_dot[n:] = forces      # dv/dt = F/m
        
        return state_dot
    
    def simulate(self, initial_state, t_span, control_func=None, excitation_func=None, dt=0.001):
        """
        Simulate the multi-layer system
        
        Args:
            initial_state: Initial state vector
            t_span: Time span [t_start, t_end] or time array
            control_func: Control function u(t, x)
            excitation_func: Base excitation function w(t)
            dt: Time step for integration
            
        Returns:
            Time array and state trajectories
        """
        if isinstance(t_span, (list, tuple)) and len(t_span) == 2:
            t = np.arange(t_span[0], t_span[1] + dt, dt)
        else:
            t = np.array(t_span)
            dt = t[1] - t[0] if len(t) > 1 else dt
        
        states = np.zeros((len(t), len(initial_state)))
        control_history = np.zeros((len(t), self.dof))
        
        states[0] = initial_state
        
        for i in range(1, len(t)):
            current_time = t[i-1]
            current_state = states[i-1]
            
            # Calculate control forces
            if control_func is not None:
                control_forces = control_func(current_time, current_state)
            else:
                control_forces = np.zeros(self.dof)
            
            control_history[i-1] = control_forces
            
            # Integration step
            def dynamics(t, x):
                return self.system_dynamics(t, x, control_forces, excitation_func)
            
            # RK4 integration
            k1 = dynamics(current_time, current_state)
            k2 = dynamics(current_time + dt/2, current_state + dt/2 * k1)
            k3 = dynamics(current_time + dt/2, current_state + dt/2 * k2)
            k4 = dynamics(current_time + dt, current_state + dt * k3)
            
            states[i] = current_state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Final control calculation
        if control_func is not None:
            control_history[-1] = control_func(t[-1], states[-1])
        
        return t, states, control_history
    
    def get_linearized_system(self, equilibrium_state=None):
        """
        Get linearized system matrices around equilibrium
        
        Args:
            equilibrium_state: Equilibrium state (default: zero state)
            
        Returns:
            A, B matrices for dx/dt = Ax + Bu
        """
        if equilibrium_state is None:
            equilibrium_state = np.zeros(self.state_dim)
        
        n = self.dof
        A = np.zeros((2*n, 2*n))
        B = np.zeros((2*n, n))
        
        # A matrix structure: [0, I; -M^-1*K, -M^-1*C]
        A[:n, n:] = np.eye(n)  # dx/dt = v
        
        # Linearized stiffness and damping matrices
        K = np.zeros((n, n))
        C = np.zeros((n, n))
        
        for i in range(n):
            # Diagonal terms (self stiffness and damping)
            K[i, i] = self.linear_stiffness[i]
            C[i, i] = self.damping[i]
            
            # Off-diagonal terms (coupling)
            if i < n - 1:
                K[i, i+1] = -self.linear_stiffness[i+1]
                K[i+1, i] = -self.linear_stiffness[i+1]
                C[i, i+1] = -self.damping[i+1]
                C[i+1, i] = -self.damping[i+1]
        
        # Mass matrix
        M = np.diag(self.masses)
        M_inv = np.linalg.inv(M)
        
        A[n:, :n] = -M_inv @ K
        A[n:, n:] = -M_inv @ C
        
        # B matrix (control input)
        B[n:, :] = M_inv
        
        return A, B
    
    def compute_lqr_gain(self, Q=None, R=None):
        """
        Compute LQR optimal gain
        
        Args:
            Q: State cost matrix
            R: Control cost matrix
            
        Returns:
            LQR gain matrix K
        """
        A, B = self.get_linearized_system()
        
        if Q is None:
            Q = np.eye(self.state_dim)
            # Higher weight on position control
            Q[:self.dof, :self.dof] *= 100
        
        if R is None:
            R = np.eye(self.dof)
        
        # Solve algebraic Riccati equation
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K
        except Exception as e:
            print(f"LQR computation failed: {e}")
            return np.zeros((self.dof, self.state_dim))
    
    def get_transfer_function_data(self, freq_range=None):
        """
        Compute frequency response data for system analysis
        
        Args:
            freq_range: Frequency range [f_min, f_max, n_points]
            
        Returns:
            Frequencies, magnitude, and phase responses
        """
        if freq_range is None:
            freq_range = [0.1, 100, 1000]
        
        f_min, f_max, n_points = freq_range
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
        
        A, B = self.get_linearized_system()
        C = np.zeros((self.dof, self.state_dim))
        C[:, :self.dof] = np.eye(self.dof)  # Output: positions
        
        magnitude_response = np.zeros((self.dof, len(frequencies)))
        phase_response = np.zeros((self.dof, len(frequencies)))
        
        for i, freq in enumerate(frequencies):
            omega = 2 * np.pi * freq
            s = 1j * omega
            
            try:
                # Transfer function: G(s) = C(sI - A)^(-1)B
                sI_minus_A = s * np.eye(A.shape[0]) - A
                G = C @ np.linalg.solve(sI_minus_A, B)
                
                for j in range(self.dof):
                    magnitude_response[j, i] = np.abs(G[j, j])
                    phase_response[j, i] = np.angle(G[j, j]) * 180 / np.pi
            except:
                magnitude_response[:, i] = 0
                phase_response[:, i] = 0
        
        return frequencies, magnitude_response, phase_response
    
    def get_parameters(self):
        """Get system parameters"""
        return {
            'n_layers': self.n_layers,
            'masses': self.masses.tolist(),
            'linear_stiffness': self.linear_stiffness.tolist(),
            'nonlinear_stiffness': self.nonlinear_stiffness.tolist(),
            'damping': self.damping.tolist(),
            'h0': self.h0,
            'a': self.a,
            'control_limit': self.control_limit
        }
    
    def set_parameters(self, params):
        """Set system parameters"""
        if 'masses' in params:
            self.masses = np.array(params['masses'])
        if 'linear_stiffness' in params:
            self.linear_stiffness = np.array(params['linear_stiffness'])
        if 'nonlinear_stiffness' in params:
            self.nonlinear_stiffness = np.array(params['nonlinear_stiffness'])
        if 'damping' in params:
            self.damping = np.array(params['damping'])
        if 'h0' in params:
            self.h0 = params['h0']
        if 'a' in params:
            self.a = params['a']
        if 'control_limit' in params:
            self.control_limit = params['control_limit']
    
    def create_gym_environment(self):
        """Create OpenAI Gym-like environment for RL training"""
        try:
            import gym
            from gym import spaces
            
            class MultiLayerEnv(gym.Env):
                def __init__(self, system):
                    super().__init__()
                    self.system = system
                    self.dt = 0.01
                    self.max_steps = 1000
                    self.current_step = 0
                    
                    # Action space: control forces
                    self.action_space = spaces.Box(
                        low=-system.control_limit,
                        high=system.control_limit,
                        shape=(system.dof,),
                        dtype=np.float32
                    )
                    
                    # Observation space: system state
                    state_bound = 10.0  # Reasonable bound for positions and velocities
                    self.observation_space = spaces.Box(
                        low=-state_bound,
                        high=state_bound,
                        shape=(system.state_dim,),
                        dtype=np.float32
                    )
                    
                    self.reset()
                
                def reset(self):
                    # Random initial conditions
                    self.state = np.random.uniform(-0.01, 0.01, self.system.state_dim)
                    self.current_step = 0
                    self.target_state = np.zeros(self.system.state_dim)
                    
                    # Random excitation parameters
                    self.excitation_freq = np.random.uniform(1.0, 10.0)
                    self.excitation_amp = np.random.uniform(0.001, 0.01)
                    
                    return self.state.astype(np.float32)
                
                def step(self, action):
                    action = np.clip(action, -self.system.control_limit, self.system.control_limit)
                    
                    # Base excitation
                    t = self.current_step * self.dt
                    excitation = lambda t: self.excitation_amp * np.sin(2 * np.pi * self.excitation_freq * t)
                    
                    # Simulate one step
                    state_dot = self.system.system_dynamics(t, self.state, action, excitation)
                    self.state = self.state + self.dt * state_dot
                    
                    # Calculate reward
                    state_error = self.state - self.target_state
                    position_error = np.sum(state_error[:self.system.dof]**2)
                    velocity_error = np.sum(state_error[self.system.dof:]**2)
                    control_cost = np.sum(action**2) * 1e-6
                    
                    reward = -(position_error + 0.1 * velocity_error + control_cost)
                    
                    # Check termination
                    self.current_step += 1
                    done = (self.current_step >= self.max_steps or 
                           np.any(np.abs(self.state[:self.system.dof]) > 0.1))
                    
                    info = {
                        'position_error': position_error,
                        'velocity_error': velocity_error,
                        'control_cost': control_cost
                    }
                    
                    return self.state.astype(np.float32), reward, done, info
                
                def render(self, mode='human'):
                    pass
            
            return MultiLayerEnv(self)
            
        except ImportError:
            print("Gym not available. Install with: pip install gym")
            return None


if __name__ == "__main__":
    # Test the multi-layer system
    print("Testing Multi-layer Vibration Isolation System")
    
    # Create 3-layer system
    system = MultiLayerSystem(n_layers=3)
    
    # Test free vibration
    initial_state = np.zeros(system.state_dim)
    initial_state[0] = 0.01  # Initial displacement of first layer
    
    # Simulation time
    t_span = [0, 5]
    
    print("Simulating free vibration...")
    t, states, control = system.simulate(initial_state, t_span)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Position responses
    plt.subplot(2, 2, 1)
    for i in range(system.dof):
        plt.plot(t, states[:, i], label=f'Layer {i+1}')
    plt.title('Position Responses')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)
    
    # Velocity responses
    plt.subplot(2, 2, 2)
    for i in range(system.dof):
        plt.plot(t, states[:, system.dof + i], label=f'Layer {i+1}')
    plt.title('Velocity Responses')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    # Phase portraits for first layer
    plt.subplot(2, 2, 3)
    plt.plot(states[:, 0], states[:, system.dof])
    plt.title('Phase Portrait - Layer 1')
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    # System energy
    plt.subplot(2, 2, 4)
    kinetic_energy = 0.5 * np.sum(system.masses * states[:, system.dof:]**2, axis=1)
    potential_energy = 0.5 * np.sum(system.linear_stiffness * states[:, :system.dof]**2, axis=1)
    total_energy = kinetic_energy + potential_energy
    
    plt.plot(t, kinetic_energy, label='Kinetic')
    plt.plot(t, potential_energy, label='Potential')
    plt.plot(t, total_energy, label='Total')
    plt.title('System Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('multilayer_system_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test LQR controller
    print("\nTesting LQR controller...")
    K = system.compute_lqr_gain()
    
    def lqr_control(t, state):
        return -K @ state
    
    print("Simulating with LQR control...")
    t, states_lqr, control_lqr = system.simulate(initial_state, t_span, lqr_control)
    
    # Compare controlled vs uncontrolled
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, states[:, 0], 'b-', label='Uncontrolled', linewidth=2)
    plt.plot(t, states_lqr[:, 0], 'r-', label='LQR Controlled', linewidth=2)
    plt.title('Layer 1 Position Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, control_lqr[:, 0], 'g-', linewidth=2)
    plt.title('Control Force - Layer 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('multilayer_lqr_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test frequency response
    print("\nComputing frequency response...")
    frequencies, magnitude, phase = system.get_transfer_function_data()
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for i in range(system.dof):
        plt.loglog(frequencies, magnitude[i], label=f'Layer {i+1}')
    plt.title('Frequency Response - Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for i in range(system.dof):
        plt.semilogx(frequencies, phase[i], label=f'Layer {i+1}')
    plt.title('Frequency Response - Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multilayer_frequency_response.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Multi-layer system test completed!")

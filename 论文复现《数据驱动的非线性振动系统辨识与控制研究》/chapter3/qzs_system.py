"""
Quasi-Zero-Stiffness (QZS) Active Vibration Isolation System
准零刚度主动隔振系统

This module implements the QZS vibration isolation system from Chapter 3 of the paper.
The system features nonlinear springs and active control for enhanced isolation performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple, Optional, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.math_utils import rk4_step
from utils.plotting import plot_time_series, plot_phase_portrait, plot_frequency_response


@dataclass
class QZSParameters:
    """
    QZS system parameters
    QZS系统参数
    """
    # Physical parameters
    m: float = 10.0           # Mass (kg) 质量
    c: float = 5.0            # Damping coefficient (Ns/m) 阻尼系数
    k1: float = 1000.0        # Linear spring stiffness (N/m) 线性弹簧刚度
    k2: float = 800.0         # Nonlinear spring parameter (N/m³) 非线性弹簧参数
    
    # QZS geometry parameters
    L: float = 0.5            # Inclined spring length (m) 倾斜弹簧长度
    theta0: float = 30.0      # Initial spring angle (degrees) 初始弹簧角度
    ks: float = 2000.0        # Inclined spring stiffness (N/m) 倾斜弹簧刚度
    
    # External excitation
    F0: float = 50.0          # Excitation amplitude (N) 激励幅值
    omega_f: float = 10.0     # Excitation frequency (rad/s) 激励频率
    
    # Control parameters
    max_control_force: float = 100.0  # Maximum control force (N) 最大控制力
    
    # Simulation parameters
    dt: float = 0.001         # Time step (s) 时间步长


class QZSSystem:
    """
    Quasi-Zero-Stiffness vibration isolation system
    准零刚度隔振系统
    
    The system equation is:
    m*ẍ + c*ẋ + k_eff(x)*x = F_excitation + F_control
    
    where k_eff(x) is the effective nonlinear stiffness
    """
    
    def __init__(self, params: Optional[QZSParameters] = None):
        """
        Initialize QZS system
        
        Args:
            params: System parameters
        """
        self.params = params if params is not None else QZSParameters()
        
        # Convert angle to radians
        self.theta0_rad = np.deg2rad(self.params.theta0)
        
        # Calculate equilibrium position and effective stiffness
        self._calculate_equilibrium()
        
        # State history for analysis
        self.time_history = []
        self.state_history = []
        self.force_history = []
        
    def _calculate_equilibrium(self):
        """Calculate equilibrium position and effective stiffness"""
        
        # Static equilibrium under gravity
        g = 9.81  # Gravity acceleration
        
        # Weight force
        W = self.params.m * g
        
        # Find equilibrium position where spring force balances weight
        # This is a simplified calculation - in practice, this might need numerical solution
        self.x_eq = W / self.params.k1
        
        # Calculate effective stiffness at equilibrium
        self.k_eff_eq = self._effective_stiffness(self.x_eq)
        
        print(f"Equilibrium position: {self.x_eq:.4f} m")
        print(f"Effective stiffness at equilibrium: {self.k_eff_eq:.2f} N/m")
    
    def _effective_stiffness(self, x: float) -> float:
        """
        Calculate effective stiffness of the QZS system
        
        Args:
            x: Displacement from equilibrium (m)
            
        Returns:
            Effective stiffness (N/m)
        """
        
        # Linear spring contribution
        k_linear = self.params.k1
        
        # Inclined springs contribution (QZS effect)
        # Geometry calculations
        h = self.params.L * np.sin(self.theta0_rad)  # Vertical projection
        l_horizontal = self.params.L * np.cos(self.theta0_rad)  # Horizontal projection
        
        # Current spring length and angle
        current_length = np.sqrt((h - x)**2 + l_horizontal**2)
        current_angle = np.arctan2(h - x, l_horizontal)
        
        # Force components from inclined springs (two springs symmetrically placed)
        spring_force = self.params.ks * (current_length - self.params.L)
        vertical_component = spring_force * np.sin(current_angle)
        
        # Total effective stiffness (derivative of total force w.r.t. displacement)
        # This includes both linear spring and nonlinear QZS springs
        k_qzs = 2 * self.params.ks * (np.sin(current_angle)**2 + 
                                     (spring_force / current_length) * np.cos(current_angle)**2)
        
        # Nonlinear stiffness term
        k_nonlinear = 3 * self.params.k2 * x**2
        
        # Total effective stiffness
        k_eff = k_linear - k_qzs + k_nonlinear
        
        return k_eff
    
    def _qzs_force(self, x: float) -> float:
        """
        Calculate the restoring force from QZS springs
        
        Args:
            x: Displacement from equilibrium (m)
            
        Returns:
            QZS spring force (N)
        """
        
        # Geometry calculations
        h = self.params.L * np.sin(self.theta0_rad)
        l_horizontal = self.params.L * np.cos(self.theta0_rad)
        
        # Current configuration
        current_length = np.sqrt((h - x)**2 + l_horizontal**2)
        current_angle = np.arctan2(h - x, l_horizontal)
        
        # Spring extension
        extension = current_length - self.params.L
        
        # Force from each inclined spring
        spring_force = self.params.ks * extension
        
        # Vertical component (two springs)
        vertical_force = 2 * spring_force * np.sin(current_angle)
        
        return vertical_force
    
    def dynamics(self, t: float, state: np.ndarray, 
                 control_force: float = 0.0, 
                 external_force: Optional[float] = None) -> np.ndarray:
        """
        System dynamics: dx/dt = f(t, x, u)
        
        Args:
            t: Time (s)
            state: State vector [x, dx/dt]
            control_force: Control force (N)
            external_force: External disturbance force (N)
            
        Returns:
            State derivative [dx/dt, d²x/dt²]
        """
        
        x, dx_dt = state
        
        # External excitation force
        if external_force is None:
            F_ext = self.params.F0 * np.sin(self.params.omega_f * t)
        else:
            F_ext = external_force
        
        # Spring forces
        # Linear spring force
        F_linear = -self.params.k1 * x
        
        # QZS spring force
        F_qzs = -self._qzs_force(x)
        
        # Nonlinear spring force
        F_nonlinear = -self.params.k2 * x**3
        
        # Damping force
        F_damping = -self.params.c * dx_dt
        
        # Total force
        F_total = F_linear + F_qzs + F_nonlinear + F_damping + F_ext + control_force
        
        # Acceleration
        d2x_dt2 = F_total / self.params.m
        
        return np.array([dx_dt, d2x_dt2])
    
    def integrate_step(self, state: np.ndarray, t: float, 
                      control_force: float = 0.0,
                      external_force: Optional[float] = None) -> np.ndarray:
        """
        Integrate one time step using RK4
        
        Args:
            state: Current state [x, dx/dt]
            t: Current time
            control_force: Control force
            external_force: External force
            
        Returns:
            Next state
        """
        
        def dynamics_wrapper(t, y):
            return self.dynamics(t, y, control_force, external_force)
        
        next_state = rk4_step(dynamics_wrapper, t, state, self.params.dt)
        
        return next_state
    
    def simulate(self, t_span: Tuple[float, float], 
                 initial_state: np.ndarray,
                 control_func: Optional[callable] = None,
                 disturbance_func: Optional[callable] = None) -> dict:
        """
        Simulate the QZS system
        
        Args:
            t_span: Time span (start, end)
            initial_state: Initial state [x0, dx0/dt]
            control_func: Control function f(t, state) -> control_force
            disturbance_func: Disturbance function f(t) -> force
            
        Returns:
            Dictionary containing simulation results
        """
        
        t_start, t_end = t_span
        dt = self.params.dt
        n_steps = int((t_end - t_start) / dt)
        
        # Initialize arrays
        time = np.linspace(t_start, t_end, n_steps + 1)
        states = np.zeros((n_steps + 1, 2))
        controls = np.zeros(n_steps + 1)
        external_forces = np.zeros(n_steps + 1)
        
        # Initial conditions
        states[0] = initial_state
        t = t_start
        
        # Simulation loop
        for i in range(n_steps):
            current_state = states[i]
            
            # Get control force
            if control_func is not None:
                control_force = control_func(time[i], current_state)
                # Saturate control force
                control_force = np.clip(control_force, 
                                      -self.params.max_control_force,
                                       self.params.max_control_force)
            else:
                control_force = 0.0
            
            # Get external disturbance
            if disturbance_func is not None:
                external_force = disturbance_func(time[i])
            else:
                external_force = None
            
            # Record forces
            controls[i] = control_force
            if external_force is None:
                external_forces[i] = self.params.F0 * np.sin(self.params.omega_f * time[i])
            else:
                external_forces[i] = external_force
            
            # Integrate one step
            states[i + 1] = self.integrate_step(current_state, time[i], 
                                              control_force, external_force)
        
        # Final values
        controls[-1] = controls[-2]
        external_forces[-1] = external_forces[-2]
        
        # Store history
        self.time_history = time
        self.state_history = states
        self.force_history = {'control': controls, 'external': external_forces}
        
        return {
            'time': time,
            'states': states,
            'displacement': states[:, 0],
            'velocity': states[:, 1],
            'control_force': controls,
            'external_force': external_forces,
            'effective_stiffness': [self._effective_stiffness(x) for x in states[:, 0]]
        }
    
    def frequency_response(self, freq_range: Tuple[float, float], 
                          n_frequencies: int = 100,
                          amplitude: float = 1.0) -> dict:
        """
        Calculate frequency response of the system
        
        Args:
            freq_range: Frequency range (start, end) in Hz
            n_frequencies: Number of frequency points
            amplitude: Excitation amplitude
            
        Returns:
            Dictionary containing frequency response data
        """
        
        frequencies = np.linspace(freq_range[0], freq_range[1], n_frequencies)
        omega_list = 2 * np.pi * frequencies
        
        # Response amplitude and phase
        magnitude = np.zeros(n_frequencies)
        phase = np.zeros(n_frequencies)
        
        for i, omega in enumerate(omega_list):
            # Temporarily set excitation frequency
            original_omega = self.params.omega_f
            original_F0 = self.params.F0
            
            self.params.omega_f = omega
            self.params.F0 = amplitude
            
            # Simulate for steady state
            t_transient = 5 * 2 * np.pi / omega  # 5 cycles for transient
            t_measurement = 10 * 2 * np.pi / omega  # 10 cycles for measurement
            
            # Simulate
            result = self.simulate(
                t_span=(0, t_transient + t_measurement),
                initial_state=np.array([0.0, 0.0])
            )
            
            # Extract steady-state response (last 10 cycles)
            idx_start = int(t_transient / self.params.dt)
            steady_displacement = result['displacement'][idx_start:]
            
            # Calculate response amplitude and phase
            magnitude[i] = np.max(steady_displacement) - np.min(steady_displacement)
            magnitude[i] /= 2  # Amplitude (not peak-to-peak)
            
            # Phase calculation (simplified)
            # In practice, you'd use FFT or correlation for accurate phase
            phase[i] = 0.0  # Placeholder
            
            # Restore original parameters
            self.params.omega_f = original_omega
            self.params.F0 = original_F0
        
        return {
            'frequencies': frequencies,
            'omega': omega_list,
            'magnitude': magnitude,
            'phase': phase,
            'magnitude_db': 20 * np.log10(magnitude / amplitude)
        }
    
    def analyze_qzs_effect(self, displacement_range: Tuple[float, float] = (-0.1, 0.1),
                          n_points: int = 1000) -> dict:
        """
        Analyze the QZS effect by plotting stiffness vs displacement
        
        Args:
            displacement_range: Range of displacements to analyze
            n_points: Number of points
            
        Returns:
            Analysis results
        """
        
        displacements = np.linspace(displacement_range[0], displacement_range[1], n_points)
        effective_stiffness = [self._effective_stiffness(x) for x in displacements]
        qzs_forces = [self._qzs_force(x) for x in displacements]
        
        # Find minimum stiffness point
        min_stiffness_idx = np.argmin(effective_stiffness)
        min_stiffness_pos = displacements[min_stiffness_idx]
        min_stiffness_val = effective_stiffness[min_stiffness_idx]
        
        return {
            'displacements': displacements,
            'effective_stiffness': effective_stiffness,
            'qzs_forces': qzs_forces,
            'min_stiffness_position': min_stiffness_pos,
            'min_stiffness_value': min_stiffness_val,
            'stiffness_reduction_ratio': min_stiffness_val / self.params.k1
        }
    
    def linearize_at_equilibrium(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the system at equilibrium for LQR controller design
        
        Returns:
            State matrix A and input matrix B
        """
        
        # State: [x, dx/dt]
        # Input: control force
        
        # A matrix (state transition)
        k_eff = self._effective_stiffness(0.0)  # At equilibrium
        
        A = np.array([
            [0.0, 1.0],
            [-k_eff / self.params.m, -self.params.c / self.params.m]
        ])
        
        # B matrix (input)
        B = np.array([
            [0.0],
            [1.0 / self.params.m]
        ])
        
        return A, B
    
    def get_natural_frequency(self) -> float:
        """
        Calculate natural frequency at equilibrium
        
        Returns:
            Natural frequency (Hz)
        """
        
        k_eff = self._effective_stiffness(0.0)
        omega_n = np.sqrt(k_eff / self.params.m)
        f_n = omega_n / (2 * np.pi)
        
        return f_n


class QZSEnvironment:
    """
    Reinforcement learning environment for QZS system control
    QZS系统强化学习环境
    """
    
    def __init__(self, qzs_system: QZSSystem, max_episode_steps: int = 1000):
        """
        Initialize RL environment
        
        Args:
            qzs_system: QZS system instance
            max_episode_steps: Maximum steps per episode
        """
        self.system = qzs_system
        self.max_episode_steps = max_episode_steps
        
        # Environment parameters
        self.max_displacement = 0.2  # Maximum allowed displacement (m)
        self.max_velocity = 2.0      # Maximum allowed velocity (m/s)
        self.target_position = 0.0   # Target position
        
        # Episode tracking
        self.current_step = 0
        self.current_time = 0.0
        self.current_state = None
        
        # Observation and action spaces (for compatibility with RL libraries)
        self.observation_space_dim = 4  # [x, dx/dt, x_target, time]
        self.action_space_dim = 1       # [control_force]
        self.action_space_high = self.system.params.max_control_force
        
    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment for new episode
        
        Args:
            initial_state: Initial state [x, dx/dt], if None use random
            
        Returns:
            Initial observation
        """
        
        self.current_step = 0
        self.current_time = 0.0
        
        if initial_state is None:
            # Random initial state within reasonable bounds
            x0 = np.random.uniform(-0.05, 0.05)
            v0 = np.random.uniform(-0.1, 0.1)
            self.current_state = np.array([x0, v0])
        else:
            self.current_state = initial_state.copy()
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment
        
        Args:
            action: Control action [control_force]
            
        Returns:
            observation, reward, done, info
        """
        
        # Extract control force and clip to limits
        control_force = np.clip(action[0], 
                               -self.system.params.max_control_force,
                                self.system.params.max_control_force)
        
        # Integrate system
        self.current_state = self.system.integrate_step(
            self.current_state, self.current_time, control_force
        )
        
        # Update time and step counter
        self.current_time += self.system.params.dt
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(control_force)
        
        # Check if episode is done
        done = self._is_done()
        
        # Info dictionary
        info = {
            'displacement': self.current_state[0],
            'velocity': self.current_state[1],
            'control_force': control_force,
            'time': self.current_time
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        
        x, v = self.current_state
        
        # Normalize state components
        x_norm = x / self.max_displacement
        v_norm = v / self.max_velocity
        target_norm = self.target_position / self.max_displacement
        time_norm = self.current_time / (self.max_episode_steps * self.system.params.dt)
        
        return np.array([x_norm, v_norm, target_norm, time_norm])
    
    def _calculate_reward(self, control_force: float) -> float:
        """
        Calculate reward for current state and action
        
        Args:
            control_force: Applied control force
            
        Returns:
            Reward value
        """
        
        x, v = self.current_state
        
        # Position error
        position_error = abs(x - self.target_position)
        
        # Velocity penalty (prefer low velocities for stability)
        velocity_penalty = v**2
        
        # Control effort penalty
        control_penalty = (control_force / self.system.params.max_control_force)**2
        
        # Distance from dangerous regions penalty
        safety_penalty = 0.0
        if abs(x) > 0.8 * self.max_displacement:
            safety_penalty = 10.0 * (abs(x) - 0.8 * self.max_displacement)**2
        
        # Reward components
        position_reward = -100.0 * position_error**2
        velocity_reward = -10.0 * velocity_penalty
        control_reward = -1.0 * control_penalty
        safety_reward = -safety_penalty
        
        # Bonus for being close to target
        if position_error < 0.01:  # Within 1cm
            stability_bonus = 10.0
        else:
            stability_bonus = 0.0
        
        total_reward = (position_reward + velocity_reward + 
                       control_reward + safety_reward + stability_bonus)
        
        return total_reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate"""
        
        x, v = self.current_state
        
        # Episode terminates if:
        # 1. Maximum steps reached
        # 2. System becomes unstable (large displacements/velocities)
        
        max_steps_reached = self.current_step >= self.max_episode_steps
        
        unstable = (abs(x) > self.max_displacement or 
                   abs(v) > self.max_velocity)
        
        return max_steps_reached or unstable


def demo_qzs_system():
    """Demonstration of QZS system capabilities"""
    
    print("QZS System Demonstration")
    print("=" * 50)
    
    # Create QZS system
    params = QZSParameters(
        m=10.0,
        c=5.0,
        k1=1000.0,
        k2=800.0,
        F0=20.0,
        omega_f=5.0
    )
    
    qzs = QZSSystem(params)
    
    # 1. Analyze QZS effect
    print("\n1. Analyzing QZS effect...")
    analysis = qzs.analyze_qzs_effect()
    
    print(f"Minimum stiffness: {analysis['min_stiffness_value']:.2f} N/m")
    print(f"Stiffness reduction ratio: {analysis['stiffness_reduction_ratio']:.3f}")
    print(f"Natural frequency: {qzs.get_natural_frequency():.2f} Hz")
    
    # 2. Free vibration simulation
    print("\n2. Free vibration simulation...")
    
    # Initial conditions
    initial_state = np.array([0.05, 0.0])  # 5cm initial displacement
    
    # Simulate without control
    result_free = qzs.simulate(
        t_span=(0, 5),
        initial_state=initial_state
    )
    
    print(f"Free vibration completed. Max displacement: {np.max(np.abs(result_free['displacement'])):.4f} m")
    
    # 3. Forced vibration simulation
    print("\n3. Forced vibration simulation...")
    
    result_forced = qzs.simulate(
        t_span=(0, 10),
        initial_state=np.array([0.0, 0.0])
    )
    
    steady_state_amplitude = np.std(result_forced['displacement'][-1000:]) * 2 * np.sqrt(2)
    print(f"Steady-state amplitude: {steady_state_amplitude:.4f} m")
    
    return qzs, result_free, result_forced


if __name__ == "__main__":
    # Run demonstration
    system, free_result, forced_result = demo_qzs_system()
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Free vibration
    plt.subplot(2, 3, 1)
    plt.plot(free_result['time'], free_result['displacement'])
    plt.title('Free Vibration - Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(free_result['displacement'], free_result['velocity'])
    plt.title('Free Vibration - Phase Portrait')
    plt.xlabel('Displacement (m)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    # Forced vibration
    plt.subplot(2, 3, 3)
    plt.plot(forced_result['time'], forced_result['displacement'])
    plt.title('Forced Vibration - Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(forced_result['time'], forced_result['external_force'])
    plt.title('External Force')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.grid(True)
    
    # Effective stiffness
    plt.subplot(2, 3, 5)
    analysis = system.analyze_qzs_effect()
    plt.plot(analysis['displacements'], analysis['effective_stiffness'])
    plt.title('Effective Stiffness vs Displacement')
    plt.xlabel('Displacement (m)')
    plt.ylabel('Stiffness (N/m)')
    plt.grid(True)
    
    # QZS force
    plt.subplot(2, 3, 6)
    plt.plot(analysis['displacements'], analysis['qzs_forces'])
    plt.title('QZS Spring Force')
    plt.xlabel('Displacement (m)')
    plt.ylabel('Force (N)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nQZS system demonstration completed!")
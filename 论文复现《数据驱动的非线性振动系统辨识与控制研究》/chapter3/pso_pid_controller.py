"""
PSO-PID Controller for QZS Active Vibration Isolation System
基于粒子群优化的PID控制器用于QZS主动隔振系统

This module implements a Particle Swarm Optimization (PSO) algorithm to tune
PID controller parameters for optimal QZS system performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from qzs_system import QZSSystem, QZSParameters
from utils.plotting import plot_time_series, plot_performance_comparison, save_figure
from utils.math_utils import rms_error, settling_time, overshoot


@dataclass
class PIDParameters:
    """PID controller parameters"""
    Kp: float = 1.0    # Proportional gain
    Ki: float = 0.1    # Integral gain
    Kd: float = 0.05   # Derivative gain
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.Kp, self.Ki, self.Kd])
    
    @classmethod
    def from_array(cls, params: np.ndarray) -> 'PIDParameters':
        """Create from numpy array"""
        return cls(Kp=params[0], Ki=params[1], Kd=params[2])


@dataclass
class PSOConfig:
    """PSO configuration parameters"""
    
    # PSO parameters
    n_particles: int = 30
    max_iterations: int = 100
    w: float = 0.9          # Inertia weight
    c1: float = 2.0         # Cognitive coefficient
    c2: float = 2.0         # Social coefficient
    
    # PID parameter bounds
    Kp_bounds: Tuple[float, float] = (0.1, 100.0)
    Ki_bounds: Tuple[float, float] = (0.01, 10.0)
    Kd_bounds: Tuple[float, float] = (0.001, 5.0)
    
    # Evaluation parameters
    simulation_time: float = 10.0
    dt: float = 0.001
    
    # Performance weights
    rms_weight: float = 1.0
    settling_weight: float = 0.5
    overshoot_weight: float = 0.3
    control_effort_weight: float = 0.1
    
    # Convergence criteria
    tolerance: float = 1e-6
    stagnation_generations: int = 10


class PIDController:
    """
    PID Controller implementation
    PID控制器实现
    """
    
    def __init__(self, params: PIDParameters, dt: float = 0.001, 
                 output_limits: Tuple[float, float] = (-100.0, 100.0)):
        """
        Initialize PID controller
        
        Args:
            params: PID parameters
            dt: Sampling time
            output_limits: Control output limits
        """
        self.params = params
        self.dt = dt
        self.output_limits = output_limits
        
        # Internal states
        self.reset()
    
    def reset(self):
        """Reset controller internal states"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.derivative = 0.0
        
    def update(self, error: float) -> float:
        """
        Update PID controller
        
        Args:
            error: Current tracking error
            
        Returns:
            Control output
        """
        
        # Proportional term
        proportional = self.params.Kp * error
        
        # Integral term
        self.integral += error * self.dt
        integral_term = self.params.Ki * self.integral
        
        # Derivative term
        self.derivative = (error - self.previous_error) / self.dt
        derivative_term = self.params.Kd * self.derivative
        
        # Total output
        output = proportional + integral_term + derivative_term
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Anti-windup: prevent integral windup
        if (output >= self.output_limits[1] and error > 0) or \
           (output <= self.output_limits[0] and error < 0):
            # Don't update integral if output is saturated
            self.integral -= error * self.dt
        
        # Update previous error
        self.previous_error = error
        
        return output
    
    def get_components(self) -> Dict[str, float]:
        """Get individual PID components for analysis"""
        error = self.previous_error
        
        return {
            'proportional': self.params.Kp * error,
            'integral': self.params.Ki * self.integral,
            'derivative': self.params.Kd * self.derivative
        }


class PSO:
    """
    Particle Swarm Optimization algorithm
    粒子群优化算法
    """
    
    def __init__(self, config: PSOConfig):
        """
        Initialize PSO
        
        Args:
            config: PSO configuration
        """
        self.config = config
        
        # Problem dimensions
        self.dim = 3  # Kp, Ki, Kd
        
        # Parameter bounds
        self.bounds = np.array([
            [config.Kp_bounds[0], config.Kp_bounds[1]],
            [config.Ki_bounds[0], config.Ki_bounds[1]],
            [config.Kd_bounds[0], config.Kd_bounds[1]]
        ])
        
        # Initialize swarm
        self._initialize_swarm()
        
        # Optimization tracking
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.convergence_history = []
        
    def _initialize_swarm(self):
        """Initialize particle swarm"""
        
        n_particles = self.config.n_particles
        
        # Initialize positions randomly within bounds
        self.positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], 
            size=(n_particles, self.dim)
        )
        
        # Initialize velocities (small random values)
        velocity_range = 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.velocities = np.random.uniform(
            -velocity_range, velocity_range,
            size=(n_particles, self.dim)
        )
        
        # Personal best positions and fitness
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.full(n_particles, np.inf)
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = np.inf
        
        # Fitness tracking
        self.current_fitness = np.full(n_particles, np.inf)
    
    def _update_velocity(self, iteration: int):
        """Update particle velocities"""
        
        # Adaptive inertia weight (decreases over time)
        w = self.config.w * (1 - iteration / self.config.max_iterations)
        
        # Random factors
        r1 = np.random.random((self.config.n_particles, self.dim))
        r2 = np.random.random((self.config.n_particles, self.dim))
        
        # Cognitive component (personal best)
        cognitive = self.config.c1 * r1 * (self.personal_best_positions - self.positions)
        
        # Social component (global best)
        social = self.config.c2 * r2 * (self.global_best_position - self.positions)
        
        # Update velocities
        self.velocities = w * self.velocities + cognitive + social
        
        # Velocity clamping
        velocity_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.velocities = np.clip(self.velocities, -velocity_max, velocity_max)
    
    def _update_position(self):
        """Update particle positions"""
        
        # Update positions
        self.positions += self.velocities
        
        # Boundary handling (reflection)
        for i in range(self.dim):
            mask_lower = self.positions[:, i] < self.bounds[i, 0]
            mask_upper = self.positions[:, i] > self.bounds[i, 1]
            
            self.positions[mask_lower, i] = self.bounds[i, 0]
            self.positions[mask_upper, i] = self.bounds[i, 1]
            
            # Reverse velocity if hitting boundary
            self.velocities[mask_lower, i] *= -0.5
            self.velocities[mask_upper, i] *= -0.5
    
    def _evaluate_fitness(self, fitness_func: Callable[[np.ndarray], float]):
        """Evaluate fitness for all particles"""
        
        for i in range(self.config.n_particles):
            # Evaluate fitness
            fitness = fitness_func(self.positions[i])
            self.current_fitness[i] = fitness
            
            # Update personal best
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.positions[i].copy()
            
            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.positions[i].copy()
    
    def optimize(self, fitness_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization
        
        Args:
            fitness_func: Fitness function to minimize
            
        Returns:
            Best position and fitness
        """
        
        print("Starting PSO optimization...")
        print(f"Particles: {self.config.n_particles}, Max iterations: {self.config.max_iterations}")
        print("-" * 60)
        
        start_time = time.time()
        stagnation_counter = 0
        
        for iteration in range(self.config.max_iterations):
            # Evaluate fitness
            self._evaluate_fitness(fitness_func)
            
            # Update velocities and positions
            self._update_velocity(iteration)
            self._update_position()
            
            # Track convergence
            best_fitness = self.global_best_fitness
            mean_fitness = np.mean(self.current_fitness)
            
            self.best_fitness_history.append(best_fitness)
            self.mean_fitness_history.append(mean_fitness)
            
            # Check for stagnation
            if len(self.best_fitness_history) > 1:
                improvement = abs(self.best_fitness_history[-2] - best_fitness)
                if improvement < self.config.tolerance:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration+1:3d} | "
                      f"Best Fitness: {best_fitness:.6f} | "
                      f"Mean Fitness: {mean_fitness:.6f} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # Early stopping
            if stagnation_counter >= self.config.stagnation_generations:
                print(f"Convergence achieved at iteration {iteration+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.1f}s")
        print(f"Best fitness: {self.global_best_fitness:.6f}")
        print(f"Best parameters: Kp={self.global_best_position[0]:.4f}, "
              f"Ki={self.global_best_position[1]:.4f}, Kd={self.global_best_position[2]:.4f}")
        
        return self.global_best_position, self.global_best_fitness


class QZSPSOOptimizer:
    """
    PSO optimizer for QZS system PID controller tuning
    QZS系统PID控制器PSO优化器
    """
    
    def __init__(self, qzs_system: QZSSystem, config: PSOConfig):
        """
        Initialize optimizer
        
        Args:
            qzs_system: QZS system to control
            config: PSO configuration
        """
        self.qzs_system = qzs_system
        self.config = config
        
        # Test scenarios for robust optimization
        self.test_scenarios = self._generate_test_scenarios()
        
        # PSO instance
        self.pso = PSO(config)
        
    def _generate_test_scenarios(self) -> List[Dict]:
        """Generate diverse test scenarios for robust controller design"""
        
        scenarios = []
        
        # Scenario 1: Stabilization with initial displacement
        scenarios.append({
            'name': 'stabilization',
            'initial_state': [0.05, 0.0],  # 5cm initial displacement
            'reference': lambda t: 0.0,     # Zero reference
            'disturbance': lambda t: 0.0,   # No disturbance
            'weight': 1.0
        })
        
        # Scenario 2: Step reference tracking
        scenarios.append({
            'name': 'step_tracking',
            'initial_state': [0.0, 0.0],
            'reference': lambda t: 0.02 if t > 2.0 else 0.0,  # 2cm step at t=2s
            'disturbance': lambda t: 0.0,
            'weight': 1.0
        })
        
        # Scenario 3: Sinusoidal reference tracking
        scenarios.append({
            'name': 'sine_tracking',
            'initial_state': [0.0, 0.0],
            'reference': lambda t: 0.01 * np.sin(2 * np.pi * 0.5 * t),  # 1cm amplitude, 0.5Hz
            'disturbance': lambda t: 0.0,
            'weight': 0.8
        })
        
        # Scenario 4: Disturbance rejection
        scenarios.append({
            'name': 'disturbance_rejection',
            'initial_state': [0.0, 0.0],
            'reference': lambda t: 0.0,
            'disturbance': lambda t: 10.0 * np.sin(2 * np.pi * 1.0 * t) if t > 3.0 else 0.0,  # 10N disturbance
            'weight': 1.2
        })
        
        # Scenario 5: Combined reference and disturbance
        scenarios.append({
            'name': 'combined',
            'initial_state': [0.02, 0.0],
            'reference': lambda t: 0.01 * np.sin(2 * np.pi * 0.3 * t),
            'disturbance': lambda t: 5.0 * np.sin(2 * np.pi * 2.0 * t),
            'weight': 1.0
        })
        
        return scenarios
    
    def _simulate_scenario(self, pid_params: np.ndarray, scenario: Dict) -> Dict[str, float]:
        """
        Simulate a single scenario with given PID parameters
        
        Args:
            pid_params: PID parameters [Kp, Ki, Kd]
            scenario: Test scenario
            
        Returns:
            Performance metrics
        """
        
        # Create PID controller
        pid = PIDController(
            PIDParameters.from_array(pid_params),
            dt=self.config.dt,
            output_limits=(-self.qzs_system.params.max_control_force,
                          self.qzs_system.params.max_control_force)
        )
        
        # Define control function
        def control_func(t, state):
            reference = scenario['reference'](t)
            error = reference - state[0]  # Position error
            return pid.update(error)
        
        # Define disturbance function
        def disturbance_func(t):
            return scenario['disturbance'](t)
        
        # Simulate
        result = self.qzs_system.simulate(
            t_span=(0, self.config.simulation_time),
            initial_state=np.array(scenario['initial_state']),
            control_func=control_func,
            disturbance_func=disturbance_func
        )
        
        # Calculate performance metrics
        time_vec = result['time']
        displacement = result['displacement']
        control_force = result['control_force']
        
        # Reference trajectory
        reference_traj = np.array([scenario['reference'](t) for t in time_vec])
        
        # Tracking error
        error = displacement - reference_traj
        
        # Performance metrics
        rms_error_val = rms_error(error)
        
        try:
            settling_time_val = settling_time(time_vec, error, tolerance=0.02)  # 2% tolerance
        except:
            settling_time_val = self.config.simulation_time  # Penalize non-settling
        
        try:
            overshoot_val = overshoot(error)
        except:
            overshoot_val = 0.0
        
        # Control effort
        control_effort = np.sum(np.abs(control_force)) * self.config.dt
        max_control = np.max(np.abs(control_force))
        
        # Stability check
        final_error = abs(error[-1])
        is_stable = final_error < 0.01 and max_control < 0.9 * self.qzs_system.params.max_control_force
        
        return {
            'rms_error': rms_error_val,
            'settling_time': settling_time_val,
            'overshoot': overshoot_val,
            'control_effort': control_effort,
            'max_control': max_control,
            'final_error': final_error,
            'is_stable': is_stable
        }
    
    def fitness_function(self, pid_params: np.ndarray) -> float:
        """
        Fitness function for PSO optimization
        
        Args:
            pid_params: PID parameters to evaluate
            
        Returns:
            Fitness value (lower is better)
        """
        
        total_fitness = 0.0
        total_weight = 0.0
        
        # Evaluate all scenarios
        for scenario in self.test_scenarios:
            try:
                metrics = self._simulate_scenario(pid_params, scenario)
                
                # Skip if unstable
                if not metrics['is_stable']:
                    return 1e6  # Large penalty for instability
                
                # Weighted fitness components
                fitness = (
                    self.config.rms_weight * metrics['rms_error'] +
                    self.config.settling_weight * metrics['settling_time'] / self.config.simulation_time +
                    self.config.overshoot_weight * metrics['overshoot'] +
                    self.config.control_effort_weight * metrics['control_effort'] / 1000.0  # Normalize
                )
                
                total_fitness += scenario['weight'] * fitness
                total_weight += scenario['weight']
                
            except Exception as e:
                print(f"Error in scenario {scenario['name']}: {e}")
                return 1e6  # Large penalty for simulation errors
        
        # Average fitness
        if total_weight > 0:
            return total_fitness / total_weight
        else:
            return 1e6
    
    def optimize(self) -> Tuple[PIDParameters, float]:
        """
        Optimize PID parameters using PSO
        
        Returns:
            Optimized PID parameters and best fitness
        """
        
        print("Optimizing PID controller for QZS system...")
        print("=" * 60)
        
        # Run PSO optimization
        best_params, best_fitness = self.pso.optimize(self.fitness_function)
        
        # Convert to PID parameters
        optimal_pid = PIDParameters.from_array(best_params)
        
        return optimal_pid, best_fitness
    
    def evaluate_controller(self, pid_params: PIDParameters) -> Dict[str, Dict]:
        """
        Evaluate controller performance on all scenarios
        
        Args:
            pid_params: PID parameters to evaluate
            
        Returns:
            Detailed performance results
        """
        
        results = {}
        
        for scenario in self.test_scenarios:
            metrics = self._simulate_scenario(pid_params.to_array(), scenario)
            results[scenario['name']] = metrics
        
        return results
    
    def plot_optimization_progress(self):
        """Plot PSO optimization progress"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Fitness evolution
        iterations = range(1, len(self.pso.best_fitness_history) + 1)
        ax1.plot(iterations, self.pso.best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
        ax1.plot(iterations, self.pso.mean_fitness_history, 'r--', linewidth=2, label='Mean Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('PSO Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Parameter evolution (final swarm distribution)
        final_positions = self.pso.positions
        param_names = ['Kp', 'Ki', 'Kd']
        
        for i, name in enumerate(param_names):
            ax2.scatter(final_positions[:, i], [i] * len(final_positions), 
                       alpha=0.6, s=50, label=f'Final {name}')
        
        # Mark optimal solution
        optimal_pos = self.pso.global_best_position
        for i, name in enumerate(param_names):
            ax2.scatter(optimal_pos[i], i, marker='*', s=200, color='red')
        
        ax2.set_xlabel('Parameter Value')
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(param_names)
        ax2.set_title('Final Parameter Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def demo_pso_pid_optimization():
    """Demonstrate PSO-PID optimization for QZS system"""
    
    print("PSO-PID Optimization Demonstration")
    print("=" * 50)
    
    # Create QZS system
    qzs_params = QZSParameters(
        m=10.0,
        c=5.0,
        k1=1000.0,
        k2=800.0,
        ks=2000.0,
        F0=20.0,
        omega_f=5.0,
        max_control_force=100.0
    )
    
    qzs_system = QZSSystem(qzs_params)
    
    # PSO configuration
    pso_config = PSOConfig(
        n_particles=25,
        max_iterations=50,  # Reduced for demo
        Kp_bounds=(1.0, 50.0),
        Ki_bounds=(0.1, 5.0),
        Kd_bounds=(0.01, 2.0),
        simulation_time=8.0,
        rms_weight=1.0,
        settling_weight=0.5,
        overshoot_weight=0.3,
        control_effort_weight=0.1
    )
    
    # Create optimizer
    optimizer = QZSPSOOptimizer(qzs_system, pso_config)
    
    # Optimize PID parameters
    optimal_pid, best_fitness = optimizer.optimize()
    
    print(f"\nOptimal PID parameters:")
    print(f"Kp = {optimal_pid.Kp:.4f}")
    print(f"Ki = {optimal_pid.Ki:.4f}")
    print(f"Kd = {optimal_pid.Kd:.4f}")
    print(f"Best fitness = {best_fitness:.6f}")
    
    # Evaluate performance
    performance = optimizer.evaluate_controller(optimal_pid)
    
    print(f"\nPerformance evaluation:")
    for scenario_name, metrics in performance.items():
        print(f"\n{scenario_name.upper()}:")
        print(f"  RMS Error: {metrics['rms_error']:.6f}")
        print(f"  Settling Time: {metrics['settling_time']:.3f}s")
        print(f"  Overshoot: {metrics['overshoot']:.3f}")
        print(f"  Max Control: {metrics['max_control']:.2f}N")
        print(f"  Stable: {metrics['is_stable']}")
    
    return optimizer, optimal_pid, performance


if __name__ == "__main__":
    # Run demonstration
    optimizer, optimal_pid, performance = demo_pso_pid_optimization()
    
    # Plot optimization progress
    fig = optimizer.plot_optimization_progress()
    plt.show()
    
    print("\nPSO-PID optimization demonstration completed!")

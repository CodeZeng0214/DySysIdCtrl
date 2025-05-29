"""
Test script for Van der Pol oscillator controllers comparison
测试脚本：比较Van der Pol振荡器的各种控制器性能
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from van_der_pol_system import VanDerPolSystem, VanDerPolEnv
from lqr_controller import LQRController
from td3_controller import TD3Controller
from sac_controller import SACController
from utils.plotting import (
    plot_phase_portrait, plot_time_series, plot_control_signal,
    plot_performance_comparison, plot_system_response, save_figure
)
from utils.data_utils import calculate_performance_metrics
from utils.math_utils import rms_error, settling_time, overshoot


class ControllerTester:
    """
    Controller testing and comparison class
    控制器测试和比较类
    """
    
    def __init__(self, system_params=None, test_params=None):
        """
        Initialize the tester
        
        Args:
            system_params: Van der Pol system parameters
            test_params: Testing parameters
        """
        # Default system parameters
        if system_params is None:
            system_params = {
                'mu': 1.0,      # Nonlinearity parameter
                'omega0': 1.0,   # Natural frequency
                'mass': 1.0,     # Mass
                'dt': 0.01       # Time step
            }
        
        # Default test parameters
        if test_params is None:
            test_params = {
                'test_duration': 20.0,    # Test duration (s)
                'initial_state': [2.0, 0.0],  # Initial conditions [x, dx/dt]
                'target_state': [0.0, 0.0],   # Target state
                'disturbance_amplitude': 0.1,  # External disturbance
                'disturbance_frequency': 0.5,  # Disturbance frequency
                'n_tests': 5,                  # Number of test runs
                'save_results': True,          # Save results to file
                'plot_results': True           # Generate plots
            }
        
        self.system_params = system_params
        self.test_params = test_params
        
        # Initialize system
        self.system = VanDerPolSystem(**system_params)
        self.env = VanDerPolEnv(self.system)
        
        # Initialize controllers
        self.controllers = {}
        self._initialize_controllers()
        
        # Results storage
        self.results = {}
        
    def _initialize_controllers(self):
        """Initialize all controllers"""
        
        # LQR Controller
        Q = np.diag([10.0, 1.0])  # State weighting matrix
        R = np.array([[1.0]])     # Control weighting matrix
        self.controllers['LQR'] = LQRController(self.system, Q, R)
        
        # TD3 Controller (load pre-trained model if available)
        state_dim = 2
        action_dim = 1
        max_action = 10.0
        
        self.controllers['TD3'] = TD3Controller(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
        )
        
        # Try to load pre-trained TD3 model
        td3_model_path = Path(__file__).parent / 'models' / 'td3_model.pth'
        if td3_model_path.exists():
            self.controllers['TD3'].load(str(td3_model_path))
            print("Loaded pre-trained TD3 model")
        else:
            print("Warning: Pre-trained TD3 model not found. Using random policy.")
        
        # SAC Controller (load pre-trained model if available)
        self.controllers['SAC'] = SACController(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True
        )
        
        # Try to load pre-trained SAC model
        sac_model_path = Path(__file__).parent / 'models' / 'sac_model.pth'
        if sac_model_path.exists():
            self.controllers['SAC'].load(str(sac_model_path))
            print("Loaded pre-trained SAC model")
        else:
            print("Warning: Pre-trained SAC model not found. Using random policy.")
    
    def run_single_test(self, controller_name, test_scenario='stabilization'):
        """
        Run a single test with specified controller
        
        Args:
            controller_name: Name of the controller to test
            test_scenario: Type of test ('stabilization', 'tracking', 'disturbance')
            
        Returns:
            Dict containing test results
        """
        controller = self.controllers[controller_name]
        
        # Test parameters
        dt = self.system_params['dt']
        duration = self.test_params['test_duration']
        steps = int(duration / dt)
        
        # Initialize arrays
        time_vec = np.linspace(0, duration, steps)
        states = np.zeros((steps, 2))
        controls = np.zeros((steps, 1))
        targets = np.zeros((steps, 2))
        
        # Initial conditions
        state = np.array(self.test_params['initial_state'])
        states[0] = state
        
        # Generate reference trajectory based on test scenario
        if test_scenario == 'stabilization':
            targets[:] = self.test_params['target_state']
        elif test_scenario == 'tracking':
            # Sinusoidal tracking
            targets[:, 0] = 0.5 * np.sin(0.5 * time_vec)
            targets[:, 1] = 0.5 * 0.5 * np.cos(0.5 * time_vec)
        elif test_scenario == 'disturbance':
            targets[:] = self.test_params['target_state']
        
        # External disturbance
        if test_scenario == 'disturbance':
            disturbance = (self.test_params['disturbance_amplitude'] * 
                          np.sin(2 * np.pi * self.test_params['disturbance_frequency'] * time_vec))
        else:
            disturbance = np.zeros(steps)
        
        # Simulation loop
        start_time = time.time()
        
        for i in range(steps - 1):
            # Get control action
            if controller_name == 'LQR':
                target = targets[i]
                control = controller.get_control(state, target)
            else:  # RL controllers (TD3, SAC)
                # Normalize state for RL controllers
                normalized_state = state / np.array([3.0, 3.0])  # Normalization factors
                control = controller.select_action(normalized_state)
                if isinstance(control, torch.Tensor):
                    control = control.cpu().numpy()
                control = control.reshape(-1)
            
            controls[i] = control
            
            # Apply control and disturbance to system
            total_force = control[0] + disturbance[i]
            state = self.system.integrate_step(state, total_force)
            states[i + 1] = state
        
        # Final control action
        controls[-1] = controls[-2]
        
        computation_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(time_vec, states, controls, targets, test_scenario)
        metrics['computation_time'] = computation_time
        
        return {
            'time': time_vec,
            'states': states,
            'controls': controls,
            'targets': targets,
            'disturbance': disturbance,
            'metrics': metrics,
            'scenario': test_scenario
        }
    
    def _calculate_metrics(self, time_vec, states, controls, targets, scenario):
        """Calculate performance metrics"""
        
        # Position and velocity errors
        pos_error = states[:, 0] - targets[:, 0]
        vel_error = states[:, 1] - targets[:, 1]
        
        # RMS errors
        pos_rms = rms_error(pos_error)
        vel_rms = rms_error(vel_error)
        total_rms = np.sqrt(pos_rms**2 + vel_rms**2)
        
        # Control effort
        control_effort = np.sum(np.abs(controls)) * (time_vec[1] - time_vec[0])
        control_rms = rms_error(controls.flatten())
        max_control = np.max(np.abs(controls))
        
        # Settling time (within 2% of final value)
        try:
            pos_settling = settling_time(time_vec, pos_error, tolerance=0.02)
            vel_settling = settling_time(time_vec, vel_error, tolerance=0.02)
            settling = max(pos_settling, vel_settling)
        except:
            settling = float('inf')
        
        # Overshoot
        try:
            pos_overshoot = overshoot(pos_error)
            vel_overshoot = overshoot(vel_error)
            max_overshoot = max(pos_overshoot, vel_overshoot)
        except:
            max_overshoot = 0.0
        
        # Stability check
        final_error = np.linalg.norm([pos_error[-1], vel_error[-1]])
        is_stable = final_error < 0.1
        
        metrics = {
            'pos_rms_error': pos_rms,
            'vel_rms_error': vel_rms,
            'total_rms_error': total_rms,
            'control_effort': control_effort,
            'control_rms': control_rms,
            'max_control': max_control,
            'settling_time': settling,
            'overshoot': max_overshoot,
            'final_error': final_error,
            'is_stable': is_stable
        }
        
        return metrics
    
    def run_comprehensive_test(self):
        """Run comprehensive tests for all controllers"""
        
        print("Running comprehensive controller tests...")
        print("=" * 60)
        
        test_scenarios = ['stabilization', 'tracking', 'disturbance']
        
        for scenario in test_scenarios:
            print(f"\nTesting scenario: {scenario.upper()}")
            print("-" * 40)
            
            self.results[scenario] = {}
            
            for controller_name in self.controllers.keys():
                print(f"Testing {controller_name} controller...")
                
                # Run multiple tests and average results
                scenario_results = []
                
                for test_run in range(self.test_params['n_tests']):
                    result = self.run_single_test(controller_name, scenario)
                    scenario_results.append(result)
                
                # Average the metrics
                averaged_metrics = self._average_metrics(scenario_results)
                
                self.results[scenario][controller_name] = {
                    'metrics': averaged_metrics,
                    'sample_result': scenario_results[0]  # Keep one full result for plotting
                }
                
                print(f"  - RMS Error: {averaged_metrics['total_rms_error']:.4f}")
                print(f"  - Control Effort: {averaged_metrics['control_effort']:.4f}")
                print(f"  - Settling Time: {averaged_metrics['settling_time']:.4f}")
                print(f"  - Stable: {averaged_metrics['is_stable']}")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        
        # Generate comparison plots
        if self.test_params['plot_results']:
            self._generate_plots()
        
        # Save results
        if self.test_params['save_results']:
            self._save_results()
        
        return self.results
    
    def _average_metrics(self, results_list):
        """Average metrics across multiple test runs"""
        
        metrics_keys = results_list[0]['metrics'].keys()
        averaged = {}
        
        for key in metrics_keys:
            if key == 'is_stable':
                # For stability, use majority vote
                stability_votes = [r['metrics'][key] for r in results_list]
                averaged[key] = sum(stability_votes) > len(stability_votes) // 2
            else:
                # For numerical metrics, use mean
                values = [r['metrics'][key] for r in results_list]
                averaged[key] = np.mean(values)
        
        return averaged
    
    def _generate_plots(self):
        """Generate comparison plots"""
        
        print("\nGenerating comparison plots...")
        
        # Create results directory
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Plot for each scenario
        for scenario in self.results.keys():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Controller Comparison - {scenario.title()} Test', fontsize=16)
            
            # Time series plots
            for controller_name in self.controllers.keys():
                result = self.results[scenario][controller_name]['sample_result']
                
                # Position
                axes[0, 0].plot(result['time'], result['states'][:, 0], 
                               label=f'{controller_name}', linewidth=2)
                axes[0, 0].plot(result['time'], result['targets'][:, 0], 
                               'k--', alpha=0.7, label='Target' if controller_name == 'LQR' else "")
                
                # Velocity
                axes[0, 1].plot(result['time'], result['states'][:, 1], 
                               label=f'{controller_name}', linewidth=2)
                axes[0, 1].plot(result['time'], result['targets'][:, 1], 
                               'k--', alpha=0.7, label='Target' if controller_name == 'LQR' else "")
                
                # Control signal
                axes[1, 0].plot(result['time'], result['controls'][:, 0], 
                               label=f'{controller_name}', linewidth=2)
                
                # Phase portrait
                axes[1, 1].plot(result['states'][:, 0], result['states'][:, 1], 
                               label=f'{controller_name}', linewidth=2)
            
            # Format subplots
            axes[0, 0].set_title('Position Response')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_title('Velocity Response')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].set_title('Control Signal')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Control Force (N)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].set_title('Phase Portrait')
            axes[1, 1].set_xlabel('Position (m)')
            axes[1, 1].set_ylabel('Velocity (m/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, results_dir / f'{scenario}_comparison.png')
            plt.close()
        
        # Performance metrics comparison
        self._plot_metrics_comparison()
        
    def _plot_metrics_comparison(self):
        """Plot performance metrics comparison"""
        
        results_dir = Path(__file__).parent / 'results'
        
        # Prepare data for plotting
        scenarios = list(self.results.keys())
        controllers = list(self.controllers.keys())
        
        metrics_to_plot = ['total_rms_error', 'control_effort', 'settling_time']
        metric_labels = ['RMS Error', 'Control Effort', 'Settling Time (s)']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Performance Metrics Comparison', fontsize=16)
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            for j, controller in enumerate(controllers):
                values = []
                for scenario in scenarios:
                    value = self.results[scenario][controller]['metrics'][metric]
                    # Handle infinite settling times
                    if metric == 'settling_time' and np.isinf(value):
                        value = self.test_params['test_duration']
                    values.append(value)
                
                axes[i].bar(x + j*width, values, width, label=controller, alpha=0.8)
            
            axes[i].set_title(metric_labels[i])
            axes[i].set_xlabel('Test Scenario')
            axes[i].set_ylabel(metric_labels[i])
            axes[i].set_xticks(x + width)
            axes[i].set_xticklabels([s.title() for s in scenarios])
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Log scale for settling time if needed
            if metric == 'settling_time':
                axes[i].set_yscale('log')
        
        plt.tight_layout()
        save_figure(fig, results_dir / 'metrics_comparison.png')
        plt.close()
        
        print(f"Plots saved to {results_dir}")
    
    def _save_results(self):
        """Save results to file"""
        
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for scenario in self.results:
            serializable_results[scenario] = {}
            for controller in self.results[scenario]:
                metrics = self.results[scenario][controller]['metrics']
                # Convert numpy types to Python types
                serializable_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, (np.integer, np.floating)):
                        serializable_metrics[key] = float(value)
                    else:
                        serializable_metrics[key] = value
                
                serializable_results[scenario][controller] = {
                    'metrics': serializable_metrics
                }
        
        # Save to JSON
        with open(results_dir / 'test_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_dir / 'test_results.json'}")
    
    def print_summary(self):
        """Print a summary of test results"""
        
        print("\n" + "=" * 80)
        print("CONTROLLER PERFORMANCE SUMMARY")
        print("=" * 80)
        
        for scenario in self.results:
            print(f"\n{scenario.upper()} TEST RESULTS:")
            print("-" * 50)
            
            # Table header
            print(f"{'Controller':<12} {'RMS Error':<12} {'Control Effort':<15} {'Settling Time':<15} {'Stable':<8}")
            print("-" * 70)
            
            for controller in self.controllers.keys():
                metrics = self.results[scenario][controller]['metrics']
                
                rms_error = metrics['total_rms_error']
                control_effort = metrics['control_effort']
                settling_time = metrics['settling_time']
                is_stable = "Yes" if metrics['is_stable'] else "No"
                
                # Format settling time
                if np.isinf(settling_time):
                    settling_str = ">20.0s"
                else:
                    settling_str = f"{settling_time:.3f}s"
                
                print(f"{controller:<12} {rms_error:<12.4f} {control_effort:<15.4f} "
                      f"{settling_str:<15} {is_stable:<8}")
        
        print("\n" + "=" * 80)


def main():
    """Main testing function"""
    
    # Test parameters
    system_params = {
        'mu': 1.0,      # Van der Pol parameter
        'omega0': 1.0,   # Natural frequency
        'mass': 1.0,     # Mass
        'dt': 0.01       # Time step
    }
    
    test_params = {
        'test_duration': 15.0,
        'initial_state': [2.0, 0.0],
        'target_state': [0.0, 0.0],
        'disturbance_amplitude': 0.2,
        'disturbance_frequency': 0.3,
        'n_tests': 3,
        'save_results': True,
        'plot_results': True
    }
    
    # Create tester
    tester = ControllerTester(system_params, test_params)
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Print summary
    tester.print_summary()
    
    return results


if __name__ == "__main__":
    results = main()
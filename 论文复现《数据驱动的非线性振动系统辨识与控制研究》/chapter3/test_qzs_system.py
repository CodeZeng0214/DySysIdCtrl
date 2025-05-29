"""
Chapter 3: Testing script for QZS system identification and control
Tests the trained RK4-PINN and PSO-PID controllers with comprehensive evaluation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import argparse

from qzs_system import QZSSystem
from rk4_pinn import RK4PINN
from pso_pid_controller import PSOPIDController
import sys
sys.path.append('..')
from utils.plotting import plot_system_response, plot_phase_portrait, plot_frequency_response
from utils.data_utils import calculate_rms_error, calculate_settling_time, calculate_overshoot
from utils.math_utils import compute_frequency_response


def test_pinn_identification(qzs_system, model_path, test_config):
    """Test RK4-PINN system identification performance"""
    print("Testing RK4-PINN system identification...")
    
    # Load trained model
    pinn = RK4PINN(
        input_dim=test_config['input_dim'],
        hidden_dims=test_config['hidden_dims'],
        dt=test_config['dt'],
        device=test_config['device']
    )
    pinn.load_model(model_path)
    
    # Generate test trajectories
    test_trajectories = []
    n_test = test_config['n_test_trajectories']
    t_span = test_config['t_span']
    t = np.linspace(0, t_span, int(t_span / test_config['dt']) + 1)
    
    for i in range(n_test):
        # Different initial conditions and forcing than training
        x0 = np.random.uniform(-0.08, 0.08)
        v0 = np.random.uniform(-0.15, 0.15)
        initial_state = np.array([x0, v0])
        
        # Test with different forcing patterns
        if i % 4 == 0:  # Chirp signal
            force = lambda t: 3.0 * np.sin(2 * np.pi * (1 + 2*t/t_span) * t)
        elif i % 4 == 1:  # Step input
            force = lambda t: 2.0 * (t >= t_span/3) * (t <= 2*t_span/3)
        elif i % 4 == 2:  # Multi-frequency
            force = lambda t: 1.5 * np.sin(2*np.pi*2*t) + 1.0 * np.sin(2*np.pi*5*t)
        else:  # Random signal
            np.random.seed(i)
            force = lambda t: np.interp(t, np.linspace(0, t_span, 50), 
                                      4.0 * np.random.randn(50))
        
        # True system response
        true_states = qzs_system.simulate(initial_state, t, force)
        
        # PINN prediction
        force_values = np.array([force(ti) for ti in t])
        pred_states = pinn.predict_trajectory(initial_state, force_values, t)
        
        test_trajectories.append({
            'true_states': true_states,
            'pred_states': pred_states,
            'force': force_values,
            'initial_state': initial_state,
            'test_type': ['chirp', 'step', 'multi_freq', 'random'][i % 4]
        })
    
    # Calculate identification errors
    identification_errors = []
    for traj in test_trajectories:
        pos_error = calculate_rms_error(traj['true_states'][:, 0], traj['pred_states'][:, 0])
        vel_error = calculate_rms_error(traj['true_states'][:, 1], traj['pred_states'][:, 1])
        identification_errors.append({
            'position_rms': pos_error,
            'velocity_rms': vel_error,
            'test_type': traj['test_type']
        })
    
    # Plot identification results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RK4-PINN System Identification Results')
    
    for i, traj in enumerate(test_trajectories[:4]):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        ax.plot(t, traj['true_states'][:, 0], 'b-', label='True', linewidth=2)
        ax.plot(t, traj['pred_states'][:, 0], 'r--', label='PINN', linewidth=2)
        ax.set_title(f'Test {i+1}: {traj["test_type"].title()}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/pinn_identification_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot phase portraits
    plt.figure(figsize=(12, 8))
    for i, traj in enumerate(test_trajectories[:4]):
        plt.subplot(2, 2, i+1)
        plt.plot(traj['true_states'][:, 0], traj['true_states'][:, 1], 'b-', 
                label='True', linewidth=2)
        plt.plot(traj['pred_states'][:, 0], traj['pred_states'][:, 1], 'r--', 
                label='PINN', linewidth=2)
        plt.title(f'{traj["test_type"].title()} - Phase Portrait')
        plt.xlabel('Position (m)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/pinn_phase_portraits.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return identification_errors, test_trajectories


def test_pso_pid_control(qzs_system, controller_path, test_config):
    """Test PSO-PID controller performance"""
    print("Testing PSO-PID controller...")
    
    # Load optimized controller parameters
    with open(controller_path, 'r') as f:
        controller_data = json.load(f)
    
    pid_params = np.array(controller_data['best_params'])
    
    # Create controller
    pso_pid = PSOPIDController(qzs_system)
    pso_pid.set_pid_params(pid_params)
    
    # Define comprehensive test scenarios
    test_scenarios = [
        {
            'name': 'Step Response (Small)',
            'reference': lambda t: np.where(t >= 1.0, 0.005, 0.0),
            'disturbance': lambda t: np.zeros_like(t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 8.0
        },
        {
            'name': 'Step Response (Large)',
            'reference': lambda t: np.where(t >= 1.0, 0.02, 0.0),
            'disturbance': lambda t: np.zeros_like(t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 10.0
        },
        {
            'name': 'Sinusoidal Tracking (Low Freq)',
            'reference': lambda t: 0.01 * np.sin(2 * np.pi * 1.0 * t),
            'disturbance': lambda t: np.zeros_like(t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 6.0
        },
        {
            'name': 'Sinusoidal Tracking (High Freq)',
            'reference': lambda t: 0.005 * np.sin(2 * np.pi * 5.0 * t),
            'disturbance': lambda t: np.zeros_like(t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 4.0
        },
        {
            'name': 'Disturbance Rejection',
            'reference': lambda t: np.zeros_like(t),
            'disturbance': lambda t: 3.0 * np.sin(2 * np.pi * 3.0 * t) * (t >= 2.0) * (t <= 6.0),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 8.0
        },
        {
            'name': 'Square Wave Tracking',
            'reference': lambda t: 0.008 * np.sign(np.sin(2 * np.pi * 0.5 * t)),
            'disturbance': lambda t: np.zeros_like(t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 8.0
        },
        {
            'name': 'Ramp Input',
            'reference': lambda t: np.minimum(0.002 * t, 0.01),
            'disturbance': lambda t: np.zeros_like(t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 8.0
        },
        {
            'name': 'Complex Scenario',
            'reference': lambda t: 0.008 * np.sin(2*np.pi*1.5*t) + 0.003 * np.sin(2*np.pi*4*t),
            'disturbance': lambda t: 1.5 * np.sin(2*np.pi*8*t) * (t >= 3.0) * (t <= 7.0),
            'initial_state': np.array([0.003, 0.0]),
            'duration': 10.0
        }
    ]
    
    # Test each scenario
    test_results = []
    
    plt.figure(figsize=(16, 20))
    
    for i, scenario in enumerate(test_scenarios):
        print(f"Testing scenario: {scenario['name']}")
        
        # Simulate controlled system
        t = np.linspace(0, scenario['duration'], int(scenario['duration'] / test_config['dt']) + 1)
        ref_signal = np.array([scenario['reference'](ti) for ti in t])
        dist_signal = np.array([scenario['disturbance'](ti) for ti in t])
        
        # Run simulation
        states, control_signal = pso_pid.simulate_control(
            scenario['initial_state'], t, scenario['reference'], scenario['disturbance']
        )
        
        # Calculate performance metrics
        error_signal = ref_signal - states[:, 0]
        rms_error = calculate_rms_error(ref_signal, states[:, 0])
        settling_time = calculate_settling_time(t, states[:, 0], ref_signal[-1])
        overshoot = calculate_overshoot(states[:, 0], ref_signal[-1])
        control_effort = np.sqrt(np.mean(control_signal**2))
        
        test_results.append({
            'scenario': scenario['name'],
            'rms_error': rms_error,
            'settling_time': settling_time,
            'overshoot': overshoot,
            'control_effort': control_effort,
            'max_error': np.max(np.abs(error_signal)),
            'steady_state_error': np.mean(np.abs(error_signal[-100:]))
        })
        
        # Plot results
        plt.subplot(4, 2, i+1)
        plt.plot(t, ref_signal, 'k--', label='Reference', linewidth=2)
        plt.plot(t, states[:, 0], 'b-', label='Response', linewidth=2)
        if np.any(dist_signal != 0):
            plt.plot(t, dist_signal/100, 'r:', label='Disturbance/100', alpha=0.7)
        plt.title(f'{scenario["name"]}\nRMS Error: {rms_error:.6f}')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/pso_pid_control_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot control signals
    plt.figure(figsize=(16, 20))
    for i, scenario in enumerate(test_scenarios):
        t = np.linspace(0, scenario['duration'], int(scenario['duration'] / test_config['dt']) + 1)
        states, control_signal = pso_pid.simulate_control(
            scenario['initial_state'], t, scenario['reference'], scenario['disturbance']
        )
        
        plt.subplot(4, 2, i+1)
        plt.plot(t, control_signal, 'g-', linewidth=2)
        plt.title(f'{scenario["name"]} - Control Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Force (N)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/pso_pid_control_signals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_results


def test_frequency_response(qzs_system, controller_path, test_config):
    """Test frequency response of the controlled system"""
    print("Testing frequency response...")
    
    # Load controller
    with open(controller_path, 'r') as f:
        controller_data = json.load(f)
    pid_params = np.array(controller_data['best_params'])
    
    # Frequency range
    frequencies = np.logspace(-1, 2, 100)  # 0.1 to 100 Hz
    
    # Calculate frequency response
    open_loop_response = []
    closed_loop_response = []
    
    for freq in frequencies:
        # Open-loop response (uncontrolled system)
        omega = 2 * np.pi * freq
        
        # Linearized system matrices (around equilibrium)
        A, B = qzs_system.get_linearized_system()
        
        # Open-loop transfer function: G(s) = C(sI - A)^(-1}B
        s = 1j * omega
        sI_minus_A = s * np.eye(A.shape[0]) - A
        try:
            G_open = np.linalg.solve(sI_minus_A, B.flatten())
            C = np.array([1, 0])  # Output position
            open_loop_mag = np.abs(C @ G_open)
        except:
            open_loop_mag = 0
        
        open_loop_response.append(open_loop_mag)
        
        # Closed-loop response with PID controller
        # For PID: C(s) = Kp + Ki/s + Kd*s
        Kp, Ki, Kd = pid_params
        pid_tf = Kp + Ki/(1j*omega) + Kd*(1j*omega)
        
        # Closed-loop: T(s) = G(s)*C(s) / (1 + G(s)*C(s))
        try:
            loop_gain = pid_tf * (C @ G_open)
            closed_loop_mag = np.abs(loop_gain / (1 + loop_gain))
        except:
            closed_loop_mag = 0
            
        closed_loop_response.append(closed_loop_mag)
    
    # Plot frequency response
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.loglog(frequencies, open_loop_response, 'b-', label='Open-loop', linewidth=2)
    plt.loglog(frequencies, closed_loop_response, 'r-', label='Closed-loop (PID)', linewidth=2)
    plt.title('Frequency Response Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.subplot(2, 1, 2)
    open_loop_phase = np.angle(np.array(open_loop_response)) * 180 / np.pi
    closed_loop_phase = np.angle(np.array(closed_loop_response)) * 180 / np.pi
    plt.semilogx(frequencies, open_loop_phase, 'b-', label='Open-loop', linewidth=2)
    plt.semilogx(frequencies, closed_loop_phase, 'r-', label='Closed-loop (PID)', linewidth=2)
    plt.title('Frequency Response Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/frequency_response_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find resonance frequency and bandwidth
    max_idx = np.argmax(closed_loop_response)
    resonance_freq = frequencies[max_idx]
    max_response = closed_loop_response[max_idx]
    
    # Find -3dB bandwidth
    half_power = max_response / np.sqrt(2)
    bandwidth_indices = np.where(np.array(closed_loop_response) >= half_power)[0]
    if len(bandwidth_indices) > 0:
        bandwidth = frequencies[bandwidth_indices[-1]] - frequencies[bandwidth_indices[0]]
    else:
        bandwidth = 0
    
    frequency_metrics = {
        'resonance_frequency': resonance_freq,
        'max_response': max_response,
        'bandwidth': bandwidth
    }
    
    print(f"Resonance frequency: {resonance_freq:.2f} Hz")
    print(f"Maximum response: {max_response:.4f}")
    print(f"Bandwidth: {bandwidth:.2f} Hz")
    
    return frequency_metrics


def main():
    parser = argparse.ArgumentParser(description='Test QZS system identification and control')
    parser.add_argument('--mode', choices=['pinn', 'pid', 'freq', 'all'], default='all',
                       help='Test mode: pinn, pid, freq, or all')
    parser.add_argument('--pinn_model', default='models/rk4_pinn_epoch_2000.pth',
                       help='Path to trained PINN model')
    parser.add_argument('--pid_model', default='models/pso_pid_controller.json',
                       help='Path to optimized PID controller')
    parser.add_argument('--n_test_trajectories', type=int, default=8,
                       help='Number of test trajectories for PINN')
    parser.add_argument('--device', default='cpu',
                       help='Device for testing (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize QZS system
    qzs_system = QZSSystem()
    
    # Test configuration
    test_config = {
        'input_dim': 3,
        'hidden_dims': [64, 64, 64],
        'dt': 0.01,
        'device': args.device,
        'n_test_trajectories': args.n_test_trajectories,
        't_span': 8.0
    }
    
    results = {}
    
    if args.mode in ['pinn', 'all']:
        if os.path.exists(args.pinn_model):
            identification_errors, test_trajectories = test_pinn_identification(
                qzs_system, args.pinn_model, test_config
            )
            results['pinn'] = {
                'identification_errors': identification_errors,
                'avg_position_error': np.mean([e['position_rms'] for e in identification_errors]),
                'avg_velocity_error': np.mean([e['velocity_rms'] for e in identification_errors])
            }
        else:
            print(f"PINN model not found: {args.pinn_model}")
    
    if args.mode in ['pid', 'all']:
        if os.path.exists(args.pid_model):
            control_results = test_pso_pid_control(qzs_system, args.pid_model, test_config)
            results['pid'] = control_results
        else:
            print(f"PID controller not found: {args.pid_model}")
    
    if args.mode in ['freq', 'all']:
        if os.path.exists(args.pid_model):
            frequency_metrics = test_frequency_response(qzs_system, args.pid_model, test_config)
            results['frequency'] = frequency_metrics
        else:
            print(f"PID controller not found for frequency test: {args.pid_model}")
    
    # Save test results
    results_path = f"results/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if 'pinn' in results:
        print(f"PINN Identification:")
        print(f"  Average Position Error: {results['pinn']['avg_position_error']:.6f}")
        print(f"  Average Velocity Error: {results['pinn']['avg_velocity_error']:.6f}")
    
    if 'pid' in results:
        print(f"\nPID Control Performance:")
        avg_rms = np.mean([r['rms_error'] for r in results['pid']])
        avg_settling = np.mean([r['settling_time'] for r in results['pid'] if r['settling_time'] is not None])
        print(f"  Average RMS Error: {avg_rms:.6f}")
        print(f"  Average Settling Time: {avg_settling:.2f} s")
    
    if 'frequency' in results:
        print(f"\nFrequency Response:")
        print(f"  Resonance Frequency: {results['frequency']['resonance_frequency']:.2f} Hz")
        print(f"  Bandwidth: {results['frequency']['bandwidth']:.2f} Hz")
    
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()

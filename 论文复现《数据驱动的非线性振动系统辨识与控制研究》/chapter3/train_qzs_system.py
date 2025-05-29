"""
Chapter 3: Training script for QZS system identification and control
Trains RK4-PINN for system identification and PSO-PID for control optimization
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
from utils.plotting import plot_training_curves, plot_system_response, plot_phase_portrait
from utils.data_utils import generate_training_data, add_noise


def train_rk4_pinn(qzs_system, config):
    """Train RK4-PINN for system identification"""
    print("Training RK4-PINN for QZS system identification...")
    
    # Initialize PINN
    pinn = RK4PINN(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        dt=config['dt'],
        device=config['device']
    )
    
    # Generate training data
    print("Generating training data...")
    n_trajectories = config['n_trajectories']
    t_span = config['t_span']
    t = np.linspace(0, t_span, int(t_span / config['dt']) + 1)
    
    trajectories = []
    initial_conditions = []
    
    for i in range(n_trajectories):
        # Random initial conditions
        x0 = np.random.uniform(-0.05, 0.05)
        v0 = np.random.uniform(-0.1, 0.1)
        initial_state = np.array([x0, v0])
        initial_conditions.append(initial_state)
        
        # Generate trajectory with random forcing
        force_amplitude = np.random.uniform(0, 5.0)
        force_freq = np.random.uniform(0.5, 10.0)
        force = lambda t: force_amplitude * np.sin(2 * np.pi * force_freq * t)
        
        # Simulate system
        states = qzs_system.simulate(initial_state, t, force)
        if config['add_noise']:
            noise_level = config['noise_level']
            states = add_noise(states, noise_level)
        
        trajectories.append({
            'states': states,
            'force': np.array([force(ti) for ti in t]),
            'initial_state': initial_state
        })
    
    # Training loop
    print("Starting PINN training...")
    losses = pinn.train(trajectories, t, config['epochs'])
    
    # Save trained model
    model_path = f"models/rk4_pinn_epoch_{config['epochs']}.pth"
    os.makedirs("models", exist_ok=True)
    pinn.save_model(model_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses['total'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(losses['physics'])
    plt.title('Physics Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(losses['data'])
    plt.title('Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/rk4_pinn_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluate identified parameters
    identified_params = pinn.get_identified_parameters()
    true_params = qzs_system.get_parameters()
    
    print("\nParameter identification results:")
    print(f"True parameters: {true_params}")
    print(f"Identified parameters: {identified_params}")
    
    param_errors = {}
    for key in true_params:
        if key in identified_params:
            error = abs(identified_params[key] - true_params[key]) / abs(true_params[key]) * 100
            param_errors[key] = error
            print(f"{key}: True={true_params[key]:.4f}, Identified={identified_params[key]:.4f}, Error={error:.2f}%")
    
    return pinn, losses, param_errors


def train_pso_pid(qzs_system, config):
    """Train PSO-PID controller"""
    print("\nTraining PSO-PID controller...")
    
    # Initialize PSO-PID controller
    pso_pid = PSOPIDController(
        qzs_system=qzs_system,
        n_particles=config['n_particles'],
        n_iterations=config['n_iterations'],
        w=config['w'],
        c1=config['c1'],
        c2=config['c2']
    )
    
    # Define test scenarios for robust optimization
    test_scenarios = [
        {
            'name': 'Step Response',
            'reference': lambda t: np.where(t >= 1.0, 0.01, 0.0),
            'disturbance': lambda t: np.zeros_like(t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 10.0
        },
        {
            'name': 'Sinusoidal Tracking',
            'reference': lambda t: 0.005 * np.sin(2 * np.pi * 2.0 * t),
            'disturbance': lambda t: np.zeros_like(t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 5.0
        },
        {
            'name': 'Disturbance Rejection',
            'reference': lambda t: np.zeros_like(t),
            'disturbance': lambda t: 2.0 * np.sin(2 * np.pi * 5.0 * t) * (t >= 2.0) * (t <= 8.0),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 10.0
        },
        {
            'name': 'Mixed Scenario',
            'reference': lambda t: 0.008 * np.sin(2 * np.pi * 1.5 * t),
            'disturbance': lambda t: 1.0 * np.sin(2 * np.pi * 8.0 * t) * (t >= 3.0),
            'initial_state': np.array([0.002, 0.0]),
            'duration': 8.0
        },
        {
            'name': 'High Frequency',
            'reference': lambda t: 0.003 * np.sin(2 * np.pi * 10.0 * t),
            'disturbance': lambda t: 0.5 * np.sin(2 * np.pi * 15.0 * t),
            'initial_state': np.array([0.0, 0.0]),
            'duration': 3.0
        }
    ]
    
    # Optimize PID parameters
    best_params, convergence_history = pso_pid.optimize(test_scenarios)
    
    # Save optimized controller
    controller_path = "models/pso_pid_controller.json"
    with open(controller_path, 'w') as f:
        json.dump({
            'best_params': best_params.tolist(),
            'convergence_history': convergence_history,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history)
    plt.title('PSO-PID Optimization Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('results/pso_pid_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nOptimized PID parameters: Kp={best_params[0]:.4f}, Ki={best_params[1]:.4f}, Kd={best_params[2]:.4f}")
    
    return pso_pid, best_params, convergence_history


def main():
    parser = argparse.ArgumentParser(description='Train QZS system identification and control')
    parser.add_argument('--mode', choices=['pinn', 'pid', 'both'], default='both',
                       help='Training mode: pinn, pid, or both')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Number of training epochs for PINN')
    parser.add_argument('--n_trajectories', type=int, default=20,
                       help='Number of trajectories for PINN training')
    parser.add_argument('--pso_iterations', type=int, default=100,
                       help='Number of PSO iterations')
    parser.add_argument('--noise_level', type=float, default=0.01,
                       help='Noise level for training data')
    parser.add_argument('--device', default='cpu',
                       help='Device for training (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize QZS system
    qzs_system = QZSSystem()
    
    # Configuration for PINN training
    pinn_config = {
        'input_dim': 3,  # [x, v, F]
        'hidden_dims': [64, 64, 64],
        'dt': 0.01,
        'device': args.device,
        'epochs': args.epochs,
        'n_trajectories': args.n_trajectories,
        't_span': 10.0,
        'add_noise': True,
        'noise_level': args.noise_level
    }
    
    # Configuration for PSO-PID
    pso_config = {
        'n_particles': 30,
        'n_iterations': args.pso_iterations,
        'w': 0.7,
        'c1': 1.5,
        'c2': 1.5
    }
    
    results = {}
    
    if args.mode in ['pinn', 'both']:
        # Train RK4-PINN
        pinn, pinn_losses, param_errors = train_rk4_pinn(qzs_system, pinn_config)
        results['pinn'] = {
            'losses': pinn_losses,
            'param_errors': param_errors
        }
    
    if args.mode in ['pid', 'both']:
        # Train PSO-PID
        pso_pid, best_params, convergence_history = train_pso_pid(qzs_system, pso_config)
        results['pso_pid'] = {
            'best_params': best_params.tolist(),
            'convergence_history': convergence_history
        }
    
    # Save training results
    results_path = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTraining completed! Results saved to {results_path}")


if __name__ == "__main__":
    main()

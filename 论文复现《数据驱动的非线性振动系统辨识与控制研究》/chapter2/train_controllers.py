"""
Training script for Van der Pol oscillator controllers
训练脚本：Van der Pol振荡器控制器训练
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from pathlib import Path
import argparse

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from van_der_pol_system import VanDerPolSystem, VanDerPolEnv
from td3_controller import TD3Controller
from sac_controller import SACController
from utils.plotting import plot_training_progress, save_figure
from utils.data_utils import ReplayBuffer


class ControllerTrainer:
    """
    Controller training class for RL algorithms
    强化学习算法控制器训练类
    """
    
    def __init__(self, config):
        """
        Initialize the trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        
        # Initialize system and environment
        self.system = VanDerPolSystem(
            mu=config['system']['mu'],
            omega0=config['system']['omega0'],
            mass=config['system']['mass'],
            dt=config['system']['dt']
        )
        
        self.env = VanDerPolEnv(self.system)
        
        # Initialize controllers
        self.controllers = {}
        self._initialize_controllers()
        
        # Training tracking
        self.training_stats = {}
        
        # Create output directories
        self.models_dir = Path(__file__).parent / 'models'
        self.results_dir = Path(__file__).parent / 'results' / 'training'
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_controllers(self):
        """Initialize RL controllers"""
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])
        
        # TD3 Controller
        if 'TD3' in self.config['algorithms']:
            self.controllers['TD3'] = TD3Controller(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                **self.config['td3_params']
            )
        
        # SAC Controller
        if 'SAC' in self.config['algorithms']:
            self.controllers['SAC'] = SACController(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                **self.config['sac_params']
            )
    
    def train_controller(self, algorithm, total_timesteps):
        """
        Train a specific controller
        
        Args:
            algorithm: Algorithm name ('TD3' or 'SAC')
            total_timesteps: Total training timesteps
            
        Returns:
            Training statistics
        """
        
        controller = self.controllers[algorithm]
        config = self.config
        
        print(f"\nTraining {algorithm} controller...")
        print(f"Total timesteps: {total_timesteps}")
        print("-" * 50)
        
        # Training statistics
        episode_rewards = []
        episode_lengths = []
        actor_losses = []
        critic_losses = []
        evaluation_rewards = []
        
        # Training loop variables
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        timestep = 0
        start_time = time.time()
        
        # Start training
        while timestep < total_timesteps:
            
            # Select action
            if timestep < config['training']['start_timesteps']:
                # Random action for initial exploration
                action = self.env.action_space.sample()
            else:
                # Select action from policy with noise
                action = controller.select_action(state)
                
                # Add exploration noise for TD3
                if algorithm == 'TD3':
                    noise = np.random.normal(0, max_action * config['training']['exploration_noise'], 
                                           size=action_dim)
                    action = np.clip(action + noise, -max_action, max_action)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            controller.replay_buffer.add(state, action, next_state, reward, done)
            
            # Update tracking variables
            state = next_state
            episode_reward += reward
            episode_length += 1
            timestep += 1
            
            # Train the agent
            if timestep >= config['training']['start_timesteps']:
                losses = controller.train()
                
                if losses is not None:
                    if algorithm == 'TD3':
                        critic_losses.append(losses.get('critic_loss', 0))
                        if 'actor_loss' in losses:
                            actor_losses.append(losses['actor_loss'])
                    elif algorithm == 'SAC':
                        critic_losses.append(losses.get('critic_loss', 0))
                        actor_losses.append(losses.get('actor_loss', 0))
            
            # End of episode
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_count += 1
                
                # Print progress
                if episode_count % config['training']['log_frequency'] == 0:
                    avg_reward = np.mean(episode_rewards[-config['training']['log_frequency']:])
                    elapsed_time = time.time() - start_time
                    
                    print(f"Episode {episode_count:6d} | "
                          f"Timestep {timestep:8d} | "
                          f"Avg Reward: {avg_reward:8.2f} | "
                          f"Time: {elapsed_time:6.1f}s")
                
                # Evaluation
                if episode_count % config['training']['eval_frequency'] == 0:
                    eval_reward = self._evaluate_controller(controller, algorithm)
                    evaluation_rewards.append(eval_reward)
                    
                    print(f"Evaluation reward: {eval_reward:.2f}")
                    
                    # Save model if it's the best so far
                    if len(evaluation_rewards) == 1 or eval_reward >= max(evaluation_rewards[:-1]):
                        model_path = self.models_dir / f'{algorithm.lower()}_best_model.pth'
                        controller.save(str(model_path))
                        print(f"Saved best model to {model_path}")
                
                # Reset environment
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        # Save final model
        final_model_path = self.models_dir / f'{algorithm.lower()}_final_model.pth'
        controller.save(str(final_model_path))
        
        # Compile training statistics
        stats = {
            'algorithm': algorithm,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'actor_losses': actor_losses,
            'critic_losses': critic_losses,
            'evaluation_rewards': evaluation_rewards,
            'total_timesteps': timestep,
            'total_episodes': episode_count,
            'training_time': time.time() - start_time
        }
        
        self.training_stats[algorithm] = stats
        
        print(f"\n{algorithm} training completed!")
        print(f"Total episodes: {episode_count}")
        print(f"Training time: {stats['training_time']:.1f}s")
        print(f"Final evaluation reward: {evaluation_rewards[-1]:.2f}")
        
        return stats
    
    def _evaluate_controller(self, controller, algorithm, n_episodes=5):
        """
        Evaluate controller performance
        
        Args:
            controller: Controller to evaluate
            algorithm: Algorithm name
            n_episodes: Number of evaluation episodes
            
        Returns:
            Average reward over evaluation episodes
        """
        
        total_reward = 0
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action without exploration noise
                action = controller.select_action(state, evaluate=True)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / n_episodes
    
    def train_all_controllers(self):
        """Train all configured controllers"""
        
        print("Starting controller training...")
        print("=" * 60)
        
        total_timesteps = self.config['training']['total_timesteps']
        
        for algorithm in self.config['algorithms']:
            if algorithm in self.controllers:
                self.train_controller(algorithm, total_timesteps)
            else:
                print(f"Warning: {algorithm} controller not initialized")
        
        # Generate training plots
        self._plot_training_results()
        
        # Save training statistics
        self._save_training_stats()
        
        print("\n" + "=" * 60)
        print("All controller training completed!")
    
    def _plot_training_results(self):
        """Generate training result plots"""
        
        print("\nGenerating training plots...")
        
        if not self.training_stats:
            print("No training statistics to plot")
            return
        
        # Episode rewards plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        for algorithm, stats in self.training_stats.items():
            
            # Smooth episode rewards
            rewards = stats['episode_rewards']
            if len(rewards) > 100:
                window = min(100, len(rewards) // 10)
                smoothed_rewards = self._smooth_curve(rewards, window)
            else:
                smoothed_rewards = rewards
            
            episodes = range(len(smoothed_rewards))
            
            # Episode rewards
            axes[0, 0].plot(episodes, smoothed_rewards, label=f'{algorithm}', linewidth=2)
            
            # Evaluation rewards
            if stats['evaluation_rewards']:
                eval_episodes = np.linspace(0, len(rewards), len(stats['evaluation_rewards']))
                axes[0, 1].plot(eval_episodes, stats['evaluation_rewards'], 
                               'o-', label=f'{algorithm}', linewidth=2, markersize=4)
            
            # Critic losses
            if stats['critic_losses']:
                critic_episodes = np.linspace(0, len(rewards), len(stats['critic_losses']))
                axes[1, 0].plot(critic_episodes, self._smooth_curve(stats['critic_losses'], 50),
                               label=f'{algorithm} Critic', linewidth=2)
            
            # Actor losses
            if stats['actor_losses']:
                actor_episodes = np.linspace(0, len(rewards), len(stats['actor_losses']))
                axes[1, 1].plot(actor_episodes, self._smooth_curve(stats['actor_losses'], 50),
                               label=f'{algorithm} Actor', linewidth=2)
        
        # Format subplots
        axes[0, 0].set_title('Episode Rewards (Smoothed)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Evaluation Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Evaluation Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        axes[1, 1].set_title('Actor Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, self.results_dir / 'training_progress.png')
        plt.close()
        
        print(f"Training plots saved to {self.results_dir}")
    
    def _smooth_curve(self, values, window):
        """Smooth a curve using moving average"""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        
        return smoothed
    
    def _save_training_stats(self):
        """Save training statistics to file"""
        
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = {}
        
        for algorithm, stats in self.training_stats.items():
            serializable_stats[algorithm] = {}
            for key, value in stats.items():
                if isinstance(value, list):
                    serializable_stats[algorithm][key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_stats[algorithm][key] = float(value)
                else:
                    serializable_stats[algorithm][key] = value
        
        # Save to JSON
        stats_path = self.results_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"Training statistics saved to {stats_path}")


def get_default_config():
    """Get default training configuration"""
    
    config = {
        'system': {
            'mu': 1.0,       # Van der Pol parameter
            'omega0': 1.0,   # Natural frequency
            'mass': 1.0,     # Mass
            'dt': 0.01       # Time step
        },
        
        'algorithms': ['TD3', 'SAC'],  # Algorithms to train
        
        'training': {
            'total_timesteps': 100000,    # Total training timesteps
            'start_timesteps': 1000,      # Random action steps before training
            'exploration_noise': 0.1,     # Exploration noise for TD3
            'batch_size': 256,            # Training batch size
            'eval_frequency': 1000,       # Evaluation frequency (episodes)
            'log_frequency': 100,         # Logging frequency (episodes)
        },
        
        'td3_params': {
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_freq': 2
        },
        
        'sac_params': {
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'automatic_entropy_tuning': True
        }
    }
    
    return config


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train Van der Pol controllers')
    parser.add_argument('--algorithm', choices=['TD3', 'SAC', 'all'], default='all',
                       help='Algorithm to train')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--mu', type=float, default=1.0,
                       help='Van der Pol parameter')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.algorithm != 'all':
        config['algorithms'] = [args.algorithm]
    
    config['training']['total_timesteps'] = args.timesteps
    config['system']['mu'] = args.mu
    
    print("Training Configuration:")
    print("-" * 30)
    print(f"Algorithms: {config['algorithms']}")
    print(f"Total timesteps: {config['training']['total_timesteps']}")
    print(f"Van der Pol parameter (μ): {config['system']['mu']}")
    print(f"System frequency (ω₀): {config['system']['omega0']}")
    
    # Create trainer and start training
    trainer = ControllerTrainer(config)
    trainer.train_all_controllers()
    
    return trainer.training_stats


if __name__ == "__main__":
    stats = main()

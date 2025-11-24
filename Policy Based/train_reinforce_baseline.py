"""
Training script for REINFORCE with Baseline on CartPole-v1
Main entry point for training the agent
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from environment.cartpole_env import CartPoleEnv
from agents.reinforce_baseline_agent import REINFORCEBaselineAgent
from utils.visualization import plot_rewards, plot_rolling_average, plot_training_summary


def train():
    """
    Main training function for REINFORCE with Baseline agent
    """
    # Initialize configuration
    config = Config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Create necessary directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.PLOT_SAVE_PATH, exist_ok=True)
    
    # Initialize environment
    env = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    print(f"Environment: {config.ENV_NAME}")
    print(f"State space: {env.input_dim}, Action space: {env.output_dim}")
    
    # Initialize agent
    agent = REINFORCEBaselineAgent(env.input_dim, env.output_dim, config)
    
    # Training metrics
    episode_rewards = []
    
    print(f"\nStarting training for {config.NUM_EPISODES} episodes...")
    print("-" * 60)
    
    # Training loop
    for episode in tqdm(range(config.NUM_EPISODES), desc="Training Progress"):
        state, _ = env.reset(seed=config.SEED + episode)
        done = False
        total_reward = 0
        step_count = 0
        
        # Reset episode memory
        agent.reset_episode()
        
        while not done and step_count < config.MAX_STEPS_PER_EPISODE:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Store reward
            agent.store_reward(reward)
            
            # Update state
            state = next_state
            step_count += 1
        
        # Train the agent at the end of episode
        policy_loss, value_loss = agent.train_step()
        
        # Store episode reward
        episode_rewards.append(total_reward)
        
        # Log progress
        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f"\nEpisode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Average Reward (last {config.LOG_INTERVAL}): {avg_reward:.2f}")
            print(f"  Steps: {step_count}")
        
        # Save model periodically
        if (episode + 1) % config.SAVE_INTERVAL == 0:
            model_path = os.path.join(config.MODEL_SAVE_PATH, f"reinforce_baseline_model_episode_{episode + 1}.pth")
            agent.save_model(model_path)
    
    # Close environment
    env.close()
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, "reinforce_baseline_model_final.pth")
    agent.save_model(final_model_path)
    
    # Generate and save plots
    print("\nGenerating training plots...")
    plot_rewards(episode_rewards, config.PLOT_SAVE_PATH, "reinforce_baseline_reward_plot.png")
    plot_rolling_average(episode_rewards, window=100, save_path=config.PLOT_SAVE_PATH, 
                        filename="reinforce_baseline_rolling_average_reward_plot.png")
    plot_training_summary(episode_rewards, config.PLOT_SAVE_PATH, "reinforce_baseline_training_summary.png")
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total Episodes: {config.NUM_EPISODES}")
    print(f"Average Reward (all episodes): {np.mean(episode_rewards):.2f}")
    print(f"Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Model saved to: {final_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()

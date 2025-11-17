"""
Training script for Double DQN on CartPole-v1
Main entry point for training the agent
"""

import os
import numpy as np
from tqdm import tqdm

from config import Config
from environment.cartpole_env import CartPoleEnv
from agents.double_dqn_agent import DoubleDQNAgent
from utils.visualization import plot_rewards, plot_rolling_average, plot_training_summary


def train():
    """
    Main training function for Double DQN agent
    """
    # Initialize configuration
    config = Config()
    
    # Create necessary directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.PLOT_SAVE_PATH, exist_ok=True)
    
    # Initialize environment
    env = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    print(f"Environment: {config.ENV_NAME}")
    print(f"State space: {env.input_dim}, Action space: {env.output_dim}")
    
    # Initialize agent
    agent = DoubleDQNAgent(env.input_dim, env.output_dim, config)
    
    # Training metrics
    episode_rewards = []
    
    print(f"\nStarting training for {config.NUM_EPISODES} episodes...")
    print("-" * 60)
    
    # Training loop
    for episode in tqdm(range(config.NUM_EPISODES), desc="Training Progress"):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.train_step()
            
            # Update state
            state = next_state
            step_count += 1
        
        # Store episode reward
        episode_rewards.append(total_reward)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Soft update target network
        agent.update_target_network()
        
        # Log progress
        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f"\nEpisode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Average Reward (last {config.LOG_INTERVAL}): {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Steps: {step_count}")
        
        # Save model periodically
        if (episode + 1) % config.SAVE_INTERVAL == 0:
            model_path = os.path.join(config.MODEL_SAVE_PATH, f"dqn_model_episode_{episode + 1}.pth")
            agent.save_model(model_path)
    
    # Close environment
    env.close()
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, "dqn_model_final.pth")
    agent.save_model(final_model_path)
    
    # Generate and save plots
    print("\nGenerating training plots...")
    plot_rewards(episode_rewards, config.PLOT_SAVE_PATH)
    plot_rolling_average(episode_rewards, window=100, save_path=config.PLOT_SAVE_PATH)
    plot_training_summary(episode_rewards, config.PLOT_SAVE_PATH)
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total Episodes: {config.NUM_EPISODES}")
    print(f"Average Reward (all episodes): {np.mean(episode_rewards):.2f}")
    print(f"Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Model saved to: {final_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()

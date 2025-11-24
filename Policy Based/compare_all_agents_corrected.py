"""
Comprehensive comparison script for Policy-Based RL Agents
Trains REINFORCE, REINFORCE with Baseline, and Actor-Critic on CartPole-v1
Generates comparison plots similar to the reference implementation
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from config import Config
from environment.cartpole_env import CartPoleEnv
from agents.reinforce_agent import REINFORCEAgent
from agents.reinforce_baseline_agent import REINFORCEBaselineAgent
from agents.actor_critic_agent import ActorCriticAgent


def smooth_curve(data, window=50):
    """Calculates rolling mean for smoother plotting."""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


def train_reinforce(config, env_name):
    """Train REINFORCE agent"""
    print(f"Training REINFORCE on {env_name}...")
    
    env = CartPoleEnv(env_name, config.RENDER_MODE)
    agent = REINFORCEAgent(env.input_dim, env.output_dim, config)
    
    scores = []
    
    for ep in range(config.NUM_EPISODES):
        state, _ = env.reset(seed=config.SEED + ep)
        done = False
        total_reward = 0
        
        agent.reset_episode()
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward)
            total_reward += reward
            state = next_state
        
        scores.append(total_reward)
        
        # Update policy
        agent.train_step()
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{config.NUM_EPISODES}, Score: {np.mean(scores[-50:]):.2f}")
    
    env.close()
    return scores


def train_reinforce_baseline(config, env_name):
    """Train REINFORCE with Baseline agent"""
    print(f"\nTraining REINFORCE with Baseline on {env_name}...")
    
    env = CartPoleEnv(env_name, config.RENDER_MODE)
    agent = REINFORCEBaselineAgent(env.input_dim, env.output_dim, config)
    
    scores = []
    
    for ep in range(config.NUM_EPISODES):
        state, _ = env.reset(seed=config.SEED + ep)
        done = False
        total_reward = 0
        
        agent.reset_episode()
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward)
            total_reward += reward
            state = next_state
        
        scores.append(total_reward)
        
        # Update policy and value network
        agent.train_step()
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{config.NUM_EPISODES}, Score: {np.mean(scores[-50:]):.2f}")
    
    env.close()
    return scores


def train_actor_critic(config, env_name):
    """Train Actor-Critic agent"""
    print(f"\nTraining Actor-Critic (Online) on {env_name}...")
    
    env = CartPoleEnv(env_name, config.RENDER_MODE)
    agent = ActorCriticAgent(env.input_dim, env.output_dim, config)
    
    scores = []
    
    for ep in range(config.NUM_EPISODES):
        state, _ = env.reset(seed=config.SEED + ep)
        done = False
        score = 0
        
        agent.reset_episode()
        
        while not done:
            action, log_prob = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            
            # Online update
            agent.train_step(state, action, reward, next_state, done, log_prob)
            
            state = next_state
        
        scores.append(score)
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{config.NUM_EPISODES}, Score: {np.mean(scores[-50:]):.2f}")
    
    env.close()
    return scores


def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Create necessary directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.PLOT_SAVE_PATH, exist_ok=True)
    
    env_name = config.ENV_NAME
    
    print("=" * 60)
    print("Training All Three Policy-Based Agents")
    print("=" * 60)
    
    # Train all three agents
    scores_reinforce = train_reinforce(config, env_name)
    scores_reinforce_baseline = train_reinforce_baseline(config, env_name)
    scores_ac = train_actor_critic(config, env_name)
    
    # Smoothing
    smooth_reinforce = smooth_curve(scores_reinforce)
    smooth_reinforce_baseline = smooth_curve(scores_reinforce_baseline)
    smooth_ac = smooth_curve(scores_ac)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(smooth_ac, label='Actor-Critic', color='blue', linewidth=2)
    plt.plot(smooth_reinforce_baseline, label='REINFORCE with Baseline', color='green', linewidth=2)
    plt.plot(smooth_reinforce, label='REINFORCE', color='red', linewidth=2, alpha=0.7)
    
    plt.title(f'Actor-Critic vs REINFORCE vs Baseline: {env_name} Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Reward (50 episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config.PLOT_SAVE_PATH, 'cartpole_comparison.png')
    plt.savefig(plot_path)
    print(f"\n✓ Plot saved to '{plot_path}'")
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"REINFORCE:")
    print(f"  Average Reward (all episodes): {np.mean(scores_reinforce):.2f}")
    print(f"  Average Reward (last 50 episodes): {np.mean(scores_reinforce[-50:]):.2f}")
    print(f"\nREINFORCE with Baseline:")
    print(f"  Average Reward (all episodes): {np.mean(scores_reinforce_baseline):.2f}")
    print(f"  Average Reward (last 50 episodes): {np.mean(scores_reinforce_baseline[-50:]):.2f}")
    print(f"\nActor-Critic:")
    print(f"  Average Reward (all episodes): {np.mean(scores_ac):.2f}")
    print(f"  Average Reward (last 50 episodes): {np.mean(scores_ac[-50:]):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

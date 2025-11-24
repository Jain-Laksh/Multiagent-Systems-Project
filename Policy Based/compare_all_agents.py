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



def train_reinforce(agent, env, config, agent_name="REINFORCE"):
    """
    Train a REINFORCE agent
    
    Args:
        agent: The REINFORCE agent
        env: The environment
        config: Configuration object
        agent_name: Name for logging
        
    Returns:
        list: Episode rewards
    """
    episode_rewards = []
    
    print(f"\nTraining {agent_name} for {config.NUM_EPISODES} episodes...")
    print("-" * 60)
    
    for episode in tqdm(range(config.NUM_EPISODES), desc=f"{agent_name} Progress"):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        agent.reset_episode()
        
        while not done and step_count < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            agent.store_reward(reward)
            
            state = next_state
            step_count += 1
        
        agent.train_step()
        episode_rewards.append(total_reward)
        
        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f"\n{agent_name} - Episode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Average Reward (last {config.LOG_INTERVAL}): {avg_reward:.2f}")
    
    return episode_rewards


def train_reinforce_baseline(agent, env, config, agent_name="REINFORCE with Baseline"):
    """
    Train a REINFORCE with Baseline agent
    
    Args:
        agent: The REINFORCE with Baseline agent
        env: The environment
        config: Configuration object
        agent_name: Name for logging
        
    Returns:
        list: Episode rewards
    """
    episode_rewards = []
    
    print(f"\nTraining {agent_name} for {config.NUM_EPISODES} episodes...")
    print("-" * 60)
    
    for episode in tqdm(range(config.NUM_EPISODES), desc=f"{agent_name} Progress"):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        agent.reset_episode()
        
        while not done and step_count < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            agent.store_reward(reward)
            
            state = next_state
            step_count += 1
        
        agent.train_step()

        episode_rewards.append(total_reward)
        
        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f"\n{agent_name} - Episode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Average Reward (last {config.LOG_INTERVAL}): {avg_reward:.2f}")
    
    return episode_rewards


def plot_comprehensive_comparison(ac_rewards, reinforce_rewards, reinforce_baseline_rewards, save_path):
    """
    Plot comprehensive comparison between Actor-Critic, REINFORCE, and REINFORCE with Baseline
    
    Args:
        ac_rewards: Episode rewards for Actor-Critic
        reinforce_rewards: Episode rewards for REINFORCE
        reinforce_baseline_rewards: Episode rewards for REINFORCE with Baseline
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    episodes = range(1, len(ac_rewards) + 1)
    
    # Plot 1: Raw rewards
    plt.figure(figsize=(14, 6))
    plt.plot(episodes, ac_rewards, alpha=0.5, label='Actor-Critic', color='blue')
    plt.plot(episodes, reinforce_rewards, alpha=0.5, label='REINFORCE', color='red')
    plt.plot(episodes, reinforce_baseline_rewards, alpha=0.5, label='REINFORCE with Baseline', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Actor-Critic vs REINFORCE vs REINFORCE with Baseline: Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'all_agents_raw_rewards.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Rolling average (100 episodes)
    window = 100
    ac_rolling = np.convolve(ac_rewards, np.ones(window)/window, mode='valid')
    reinforce_rolling = np.convolve(reinforce_rewards, np.ones(window)/window, mode='valid')
    reinforce_baseline_rolling = np.convolve(reinforce_baseline_rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(14, 6))
    plt.plot(range(window, len(ac_rewards) + 1), ac_rolling, label='Actor-Critic', color='blue', linewidth=2)
    plt.plot(range(window, len(reinforce_rewards) + 1), reinforce_rolling, label='REINFORCE', color='red', linewidth=2)
    plt.plot(range(window, len(reinforce_baseline_rewards) + 1), reinforce_baseline_rolling, 
             label='REINFORCE with Baseline', color='green', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel(f'Rolling Average Reward ({window} episodes)')
    plt.title('Actor-Critic vs REINFORCE vs REINFORCE with Baseline: Rolling Average Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'all_agents_rolling_average.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Summary statistics
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Mean rewards over time (every 100 episodes)
    interval = 100
    ac_means = [np.mean(ac_rewards[i:i+interval]) for i in range(0, len(ac_rewards), interval)]
    reinforce_means = [np.mean(reinforce_rewards[i:i+interval]) for i in range(0, len(reinforce_rewards), interval)]
    reinforce_baseline_means = [np.mean(reinforce_baseline_rewards[i:i+interval]) 
                                for i in range(0, len(reinforce_baseline_rewards), interval)]
    interval_labels = [f"{i+1}-{min(i+interval, len(ac_rewards))}" for i in range(0, len(ac_rewards), interval)]
    
    x = np.arange(len(ac_means))
    width = 0.25
    
    axes[0].bar(x - width, ac_means, width, label='Actor-Critic', color='blue', alpha=0.7)
    axes[0].bar(x, reinforce_means, width, label='REINFORCE', color='red', alpha=0.7)
    axes[0].bar(x + width, reinforce_baseline_means, width, label='REINFORCE with Baseline', color='green', alpha=0.7)
    axes[0].set_xlabel('Episode Range')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Mean Rewards per 100 Episodes')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(interval_labels, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot comparison
    axes[1].boxplot([ac_rewards, reinforce_rewards, reinforce_baseline_rewards], 
                    labels=['Actor-Critic', 'REINFORCE', 'REINFORCE\nwith Baseline'])
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Reward Distribution Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'all_agents_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plots saved to {save_path}")


def main():
    """Main comparison function"""
    config = Config()
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.PLOT_SAVE_PATH, exist_ok=True)
    
    # Initialize environments
    print("Initializing environments...")
    env_ac = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    env_reinforce = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    env_reinforce_baseline = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    
    # Initialize agents
    print("Initializing agents...")
    ac_agent = ActorCriticAgent(env_ac.input_dim, env_ac.output_dim, config)
    reinforce_agent = REINFORCEAgent(env_reinforce.input_dim, env_reinforce.output_dim, config)
    reinforce_baseline_agent = REINFORCEBaselineAgent(env_reinforce_baseline.input_dim, 
                                                       env_reinforce_baseline.output_dim, config)
    
    # Train Actor-Critic
    ac_rewards = train_actor_critic(ac_agent, env_ac, config, "Actor-Critic")
    ac_agent.save_model(os.path.join(config.MODEL_SAVE_PATH, "actor_critic_comparison_final.pth"))
    env_ac.close()
    
    # Train REINFORCE
    reinforce_rewards = train_reinforce(reinforce_agent, env_reinforce, config, "REINFORCE")
    reinforce_agent.save_model(os.path.join(config.MODEL_SAVE_PATH, "reinforce_comparison_final.pth"))
    env_reinforce.close()
    
    # Train REINFORCE with Baseline
    reinforce_baseline_rewards = train_reinforce_baseline(reinforce_baseline_agent, env_reinforce_baseline, 
                                                          config, "REINFORCE with Baseline")
    reinforce_baseline_agent.save_model(os.path.join(config.MODEL_SAVE_PATH, "reinforce_baseline_comparison_final.pth"))
    env_reinforce_baseline.close()
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comprehensive_comparison(ac_rewards, reinforce_rewards, reinforce_baseline_rewards, config.PLOT_SAVE_PATH)
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\nActor-Critic Performance:")
    print(f"  Average Reward (all episodes): {np.mean(ac_rewards):.2f} ± {np.std(ac_rewards):.2f}")
    print(f"  Average Reward (last 100 episodes): {np.mean(ac_rewards[-100:]):.2f}")
    print(f"  Max Reward: {np.max(ac_rewards):.2f}")
    
    print("\nREINFORCE Performance:")
    print(f"  Average Reward (all episodes): {np.mean(reinforce_rewards):.2f} ± {np.std(reinforce_rewards):.2f}")
    print(f"  Average Reward (last 100 episodes): {np.mean(reinforce_rewards[-100:]):.2f}")
    print(f"  Max Reward: {np.max(reinforce_rewards):.2f}")
    
    print("\nREINFORCE with Baseline Performance:")
    print(f"  Average Reward (all episodes): {np.mean(reinforce_baseline_rewards):.2f} ± {np.std(reinforce_baseline_rewards):.2f}")
    print(f"  Average Reward (last 100 episodes): {np.mean(reinforce_baseline_rewards[-100:]):.2f}")
    print(f"  Max Reward: {np.max(reinforce_baseline_rewards):.2f}")
    
    # Determine winner
    performances = {
        'Actor-Critic': np.mean(ac_rewards[-100:]),
        'REINFORCE': np.mean(reinforce_rewards[-100:]),
        'REINFORCE with Baseline': np.mean(reinforce_baseline_rewards[-100:])
    }
    
    winner = max(performances, key=performances.get)
    
    print("\n" + "-" * 80)
    print(f"Best performing agent (last 100 episodes): {winner} ({performances[winner]:.2f})")
    print("=" * 80)


if __name__ == "__main__":
    main()

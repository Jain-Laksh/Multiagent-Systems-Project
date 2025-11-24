"""
Comprehensive comparison script for Actor-Critic, REINFORCE, and REINFORCE with Baseline
Trains all three agents and compares their performance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from environment.cartpole_env import CartPoleEnv
from agents.actor_critic_agent import ActorCriticAgent
from agents.reinforce_agent import REINFORCEAgent
from agents.reinforce_baseline_agent import REINFORCEBaselineAgent


def train_actor_critic(agent, env, config, agent_name="Actor-Critic"):
    """
    Train an Actor-Critic agent
    
    Args:
        agent: The Actor-Critic agent
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
        
        while not done and step_count < config.MAX_STEPS_PER_EPISODE:
            action, log_prob, entropy = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            agent.train_step(state, action, reward, next_state, done, log_prob, entropy)
            
            state = next_state
            step_count += 1
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f"\n{agent_name} - Episode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Average Reward (last {config.LOG_INTERVAL}): {avg_reward:.2f}")
    
    return episode_rewards


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

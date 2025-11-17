"""
Comprehensive comparison script for DQN, Double DQN, and SARSA
Trains all three agents and compares their performance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from environment.cartpole_env import CartPoleEnv
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.sarsa_agent import SARSAAgent


def train_dqn_based_agent(agent, env, config, agent_name="Agent"):
    """
    Train a DQN or Double DQN agent
    
    Args:
        agent: The agent to train (DQN or Double DQN)
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
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        agent.update_target_network()
        
        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f"\n{agent_name} - Episode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Average Reward (last {config.LOG_INTERVAL}): {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    return episode_rewards


def train_sarsa_agent(agent, env, config, agent_name="SARSA"):
    """
    Train a SARSA agent (on-policy)
    
    Args:
        agent: The SARSA agent
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
        
        # Select initial action (on-policy)
        action = agent.select_action(state, training=True)
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Select next action (on-policy)
            next_action = agent.select_action(next_state, training=True)
            
            # SARSA update
            agent.train_step(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f"\n{agent_name} - Episode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Average Reward (last {config.LOG_INTERVAL}): {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    return episode_rewards


def plot_comprehensive_comparison(dqn_rewards, double_dqn_rewards, sarsa_rewards, save_path):
    """
    Plot comprehensive comparison between DQN, Double DQN, and SARSA
    
    Args:
        dqn_rewards: Episode rewards for DQN
        double_dqn_rewards: Episode rewards for Double DQN
        sarsa_rewards: Episode rewards for SARSA
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    episodes = range(1, len(dqn_rewards) + 1)
    
    # Plot 1: Raw rewards
    plt.figure(figsize=(14, 6))
    plt.plot(episodes, dqn_rewards, alpha=0.5, label='DQN', color='blue')
    plt.plot(episodes, double_dqn_rewards, alpha=0.5, label='Double DQN', color='red')
    plt.plot(episodes, sarsa_rewards, alpha=0.5, label='SARSA', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN vs Double DQN vs SARSA: Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'all_agents_raw_rewards.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Rolling average (100 episodes)
    window = 100
    dqn_rolling = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
    double_dqn_rolling = np.convolve(double_dqn_rewards, np.ones(window)/window, mode='valid')
    sarsa_rolling = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(14, 6))
    plt.plot(range(window, len(dqn_rewards) + 1), dqn_rolling, label='DQN', color='blue', linewidth=2)
    plt.plot(range(window, len(double_dqn_rewards) + 1), double_dqn_rolling, label='Double DQN', color='red', linewidth=2)
    plt.plot(range(window, len(sarsa_rewards) + 1), sarsa_rolling, label='SARSA', color='green', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel(f'Rolling Average Reward ({window} episodes)')
    plt.title('DQN vs Double DQN vs SARSA: Rolling Average Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'all_agents_rolling_average.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Summary statistics
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Mean rewards over time (every 100 episodes)
    interval = 100
    dqn_means = [np.mean(dqn_rewards[i:i+interval]) for i in range(0, len(dqn_rewards), interval)]
    double_dqn_means = [np.mean(double_dqn_rewards[i:i+interval]) for i in range(0, len(double_dqn_rewards), interval)]
    sarsa_means = [np.mean(sarsa_rewards[i:i+interval]) for i in range(0, len(sarsa_rewards), interval)]
    interval_labels = [f"{i+1}-{min(i+interval, len(dqn_rewards))}" for i in range(0, len(dqn_rewards), interval)]
    
    x = np.arange(len(dqn_means))
    width = 0.25
    
    axes[0].bar(x - width, dqn_means, width, label='DQN', color='blue', alpha=0.7)
    axes[0].bar(x, double_dqn_means, width, label='Double DQN', color='red', alpha=0.7)
    axes[0].bar(x + width, sarsa_means, width, label='SARSA', color='green', alpha=0.7)
    axes[0].set_xlabel('Episode Range')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Mean Rewards per 100 Episodes')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(interval_labels, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot comparison
    axes[1].boxplot([dqn_rewards, double_dqn_rewards, sarsa_rewards], 
                    labels=['DQN', 'Double DQN', 'SARSA'])
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
    env_dqn = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    env_double_dqn = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    env_sarsa = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    
    # Initialize agents
    print("Initializing agents...")
    dqn_agent = DQNAgent(env_dqn.input_dim, env_dqn.output_dim, config)
    double_dqn_agent = DoubleDQNAgent(env_double_dqn.input_dim, env_double_dqn.output_dim, config)
    sarsa_agent = SARSAAgent(env_sarsa.input_dim, env_sarsa.output_dim, config)
    
    # Train DQN
    dqn_rewards = train_dqn_based_agent(dqn_agent, env_dqn, config, "DQN")
    dqn_agent.save_model(os.path.join(config.MODEL_SAVE_PATH, "dqn_comparison_final.pth"))
    env_dqn.close()
    
    # Train Double DQN
    double_dqn_rewards = train_dqn_based_agent(double_dqn_agent, env_double_dqn, config, "Double DQN")
    double_dqn_agent.save_model(os.path.join(config.MODEL_SAVE_PATH, "double_dqn_comparison_final.pth"))
    env_double_dqn.close()
    
    # Train SARSA
    sarsa_rewards = train_sarsa_agent(sarsa_agent, env_sarsa, config, "SARSA")
    sarsa_agent.save_model(os.path.join(config.MODEL_SAVE_PATH, "sarsa_comparison_final.pth"))
    env_sarsa.close()
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comprehensive_comparison(dqn_rewards, double_dqn_rewards, sarsa_rewards, config.PLOT_SAVE_PATH)
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\nDQN Performance:")
    print(f"  Average Reward (all episodes): {np.mean(dqn_rewards):.2f} ± {np.std(dqn_rewards):.2f}")
    print(f"  Average Reward (last 100 episodes): {np.mean(dqn_rewards[-100:]):.2f}")
    print(f"  Max Reward: {np.max(dqn_rewards):.2f}")
    
    print("\nDouble DQN Performance:")
    print(f"  Average Reward (all episodes): {np.mean(double_dqn_rewards):.2f} ± {np.std(double_dqn_rewards):.2f}")
    print(f"  Average Reward (last 100 episodes): {np.mean(double_dqn_rewards[-100:]):.2f}")
    print(f"  Max Reward: {np.max(double_dqn_rewards):.2f}")
    
    print("\nSARSA Performance:")
    print(f"  Average Reward (all episodes): {np.mean(sarsa_rewards):.2f} ± {np.std(sarsa_rewards):.2f}")
    print(f"  Average Reward (last 100 episodes): {np.mean(sarsa_rewards[-100:]):.2f}")
    print(f"  Max Reward: {np.max(sarsa_rewards):.2f}")
    
    # Determine winner
    performances = {
        'DQN': np.mean(dqn_rewards[-100:]),
        'Double DQN': np.mean(double_dqn_rewards[-100:]),
        'SARSA': np.mean(sarsa_rewards[-100:])
    }
    
    winner = max(performances, key=performances.get)
    
    print("\n" + "-" * 80)
    print(f"Best performing agent (last 100 episodes): {winner} ({performances[winner]:.2f})")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Visualization utilities for plotting training metrics
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_rewards(episode_rewards, save_path="plots", filename="reward_plot.png"):
    """
    Plot episode vs reward
    
    Args:
        episode_rewards (list): List of rewards per episode
        save_path (str): Directory to save the plot
        filename (str): Filename for the saved plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    x = [i + 1 for i in range(len(episode_rewards))]
    y = episode_rewards
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.title("Episode vs Reward")
    plt.grid(True, alpha=0.3)
    
    # Set y-axis ticks
    max_reward = max(episode_rewards) if episode_rewards else 500
    plt.yticks(np.arange(0, max_reward + 100, 100))
    
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved reward plot to {save_file}")
    plt.close()


def plot_rolling_average(episode_rewards, window=100, save_path="plots", 
                         filename="rolling_average_reward_plot.png"):
    """
    Plot rolling average of rewards
    
    Args:
        episode_rewards (list): List of rewards per episode
        window (int): Window size for rolling average
        save_path (str): Directory to save the plot
        filename (str): Filename for the saved plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    pd.Series(episode_rewards).rolling(window).mean().plot(color='red', linewidth=2)
    plt.xlabel('Episode Number')
    plt.ylabel('Rolling Average Reward')
    plt.title(f'Episode vs Average Reward (Window={window})')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis ticks
    max_reward = max(episode_rewards) if episode_rewards else 500
    plt.yticks(np.arange(0, max_reward + 100, 100))
    
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved rolling average plot to {save_file}")
    plt.close()


def plot_training_summary(episode_rewards, save_path="plots", 
                         filename="training_summary.png"):
    """
    Plot comprehensive training summary with multiple subplots
    
    Args:
        episode_rewards (list): List of rewards per episode
        save_path (str): Directory to save the plot
        filename (str): Filename for the saved plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Episode rewards
    x = [i + 1 for i in range(len(episode_rewards))]
    axes[0].plot(x, episode_rewards, alpha=0.6, label='Episode Reward')
    axes[0].set_xlabel('Episode Number')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode vs Reward')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Rolling average
    pd.Series(episode_rewards).rolling(100).mean().plot(
        ax=axes[1], color='red', linewidth=2, label='Rolling Average (100 episodes)')
    axes[1].set_xlabel('Episode Number')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Rolling Average Reward')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved training summary to {save_file}")
    plt.close()

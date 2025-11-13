import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def smooth(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')


def plot_training_curves(rewards_dict, config, save_dir=None):
    figsize = config.get('figsize', (12, 6))
    grid = config.get('grid', True)
    title = config.get('title', 'Training Curves')
    smooth_window = config.get('smooth_window', 10)
    
    styles = {
        'Actor-Critic': {'color': '#2E86AB', 'linewidth': 2.5, 'alpha': 0.9},
        'REINFORCE': {'color': '#A23B72', 'linewidth': 2.5, 'alpha': 0.9},
        'REINFORCE+Baseline': {'color': '#F18F01', 'linewidth': 2.5, 'alpha': 0.9}
    }
    
    fig = plt.figure(figsize=figsize)
    
    for name, rewards in rewards_dict.items():
        style = styles.get(name, {'color': 'gray', 'linewidth': 2, 'alpha': 0.9})
        smoothed = smooth(rewards, smooth_window)
        plt.plot(smoothed, label=f'{name} (smoothed)', **style)
    
    plt.xlabel('Episode', fontsize=13)
    plt.ylabel('Episode Return', fontsize=13)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(grid, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_curves.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved training curves to {save_path}')
    
    plt.show()
    plt.close(fig)


def print_final_results(rewards_dict, window=100):
    print(f'\nFinal Results (mean last {window} episodes):')
    print('-' * 50)
    
    results = []
    for name, rewards in rewards_dict.items():
        avg_reward = np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards)
        results.append((name, avg_reward))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, avg_reward) in enumerate(results, 1):
        print(f'{i}. {name:20s}: {avg_reward:>7.3f}')


def plot_comparative_analysis(rewards_dict, config, save_dir=None):
    smooth_window = config.get('smooth_window', 10)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'Actor-Critic': '#2E86AB',
        'REINFORCE': '#A23B72',
        'REINFORCE+Baseline': '#F18F01'
    }
    
    ax1 = axes[0, 0]
    for name, rewards in rewards_dict.items():
        smoothed = smooth(rewards, smooth_window)
        ax1.plot(smoothed, label=name, color=colors.get(name, 'gray'), linewidth=2.5, alpha=0.9)
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Episode Return', fontsize=11)
    ax1.set_title('Training Curves (Smoothed)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for name, rewards in rewards_dict.items():
        lightly_smoothed = smooth(rewards, max(3, smooth_window // 3))
        ax2.plot(lightly_smoothed, label=name, color=colors.get(name, 'gray'), linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Episode Return', fontsize=11)
    ax2.set_title('Training Curves (Less Smoothing)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    window = 50
    for name, rewards in rewards_dict.items():
        running_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        ax3.plot(running_avg, label=name, color=colors.get(name, 'gray'), linewidth=2.5, alpha=0.9)
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Running Average Return', fontsize=11)
    ax3.set_title(f'Running Average (window={window})', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    final_rewards = []
    labels = []
    for name, rewards in rewards_dict.items():
        final_rewards.append(rewards[-100:])
        labels.append(name)
    
    bp = ax4.boxplot(final_rewards, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(label, 'gray'))
        patch.set_alpha(0.6)
    
    ax4.set_ylabel('Episode Return', fontsize=11)
    ax4.set_title('Final Performance Distribution (Last 100 Episodes)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'comparative_analysis.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved comparative analysis to {save_path}')
    
    plt.show()
    plt.close(fig)

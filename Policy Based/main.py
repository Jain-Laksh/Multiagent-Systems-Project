import random
import numpy as np
import torch
import os

from config import ENV_CONFIG, AC_CONFIG, REINFORCE_CONFIG, REINFORCE_BASELINE_CONFIG, TRAINING_CONFIG, VIS_CONFIG
from environment import GridWorld
from agents import ActorCriticAgent, REINFORCEAgent, REINFORCEWithBaselineAgent
from utils.visualization import plot_training_curves, print_final_results, plot_comparative_analysis


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_agent(agent, agent_name, n_episodes, log_interval):
    rewards = []
    print(f'\nTraining {agent_name}...')
    print('-' * 50)
    
    for ep in range(1, n_episodes + 1):
        r = agent.train_episode()
        rewards.append(r)
        
        if ep % log_interval == 0:
            avg_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            print(f'Episode {ep:>4}/{n_episodes}  |  Avg Reward (last 50): {avg_reward:>7.3f}')
    
    return rewards


def run_experiment():
    set_random_seeds(TRAINING_CONFIG['seed'])
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f'\n{"="*50}')
    print(f'Plots will be saved to: {plots_dir}')
    print(f'{"="*50}')
    
    env = GridWorld(**ENV_CONFIG)
    print('\nEnvironment Configuration:')
    print(f'  Grid Size: {ENV_CONFIG["n"]}x{ENV_CONFIG["n"]}')
    print(f'  Start: {ENV_CONFIG["start"]}, Goal: {ENV_CONFIG["goal"]}')
    print(f'  Traps: {ENV_CONFIG["traps"]}')
    print(f'  Max Steps: {ENV_CONFIG["max_steps"]}')
    
    ac_agent = ActorCriticAgent(env, **AC_CONFIG)
    rf_agent = REINFORCEAgent(env, **REINFORCE_CONFIG)
    rfb_agent = REINFORCEWithBaselineAgent(env, **REINFORCE_BASELINE_CONFIG)
    
    n_episodes = TRAINING_CONFIG['n_episodes']
    log_interval = TRAINING_CONFIG['log_interval']
    
    ac_rewards = train_agent(ac_agent, 'Actor-Critic', n_episodes, log_interval)
    rf_rewards = train_agent(rf_agent, 'REINFORCE', n_episodes, log_interval)
    rfb_rewards = train_agent(rfb_agent, 'REINFORCE+Baseline', n_episodes, log_interval)
    
    rewards_dict = {
        'Actor-Critic': ac_rewards,
        'REINFORCE': rf_rewards,
        'REINFORCE+Baseline': rfb_rewards
    }
    
    print('\n' + '=' * 50)
    print_final_results(rewards_dict, TRAINING_CONFIG['eval_window'])
    print('=' * 50)
    
    vis_config = {**VIS_CONFIG, 'smooth_window': TRAINING_CONFIG['smooth_window']}
    print('\nGenerating training curves...')
    plot_training_curves(rewards_dict, vis_config, save_dir=plots_dir)
    
    print('\nGenerating comprehensive comparative analysis...')
    plot_comparative_analysis(rewards_dict, vis_config, save_dir=plots_dir)
    
    print(f'\n{"="*50}')
    print(f'✓ All plots saved successfully!')
    print(f'  Location: {plots_dir}')
    print(f'  Files: training_curves.png, comparative_analysis.png')
    print(f'{"="*50}')
    
    return rewards_dict


def main():
    print('=' * 50)
    print('Actor-Critic vs REINFORCE on GridWorld')
    print('=' * 50)
    
    try:
        run_experiment()
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user.')
    except Exception as e:
        print(f'\n\nError during training: {e}')
        raise


if __name__ == '__main__':
    main()

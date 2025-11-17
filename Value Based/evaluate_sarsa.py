"""
Evaluation script for trained SARSA agent
Tests the agent's performance on CartPole-v1
"""

import os
import time
import numpy as np
import torch

from config import Config
from environment.cartpole_env import CartPoleEnv
from agents.sarsa_agent import SARSAAgent


def evaluate(model_path=None, render=True, num_episodes=10):
    """
    Evaluate a trained SARSA agent
    
    Args:
        model_path (str): Path to the saved model. If None, uses the final model.
        render (bool): Whether to render the environment
        num_episodes (int): Number of episodes to evaluate
    """
    # Initialize configuration
    config = Config()
    
    # Set model path
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_PATH, "sarsa_model_final.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train_sarsa.py")
        return
    
    # Initialize environment with rendering if requested
    render_mode = "human" if render else "rgb_array"
    env = CartPoleEnv(config.ENV_NAME, render_mode)
    
    # Initialize agent
    agent = SARSAAgent(env.input_dim, env.output_dim, config)
    
    # Load the trained model
    agent.load_model(model_path)
    print(f"\nLoaded model from: {model_path}")
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    print("-" * 60)
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            
            while not done:
                if render:
                    env.render()
                    time.sleep(0.01)  # Small delay for better visualization
                
                # Select action (greedy, no exploration)
                action = agent.select_action(state, training=False)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1
                
                state = next_state
                
                # Print step information
                if render:
                    print(f"Step {step_count}: Action={action}, Reward={reward:.2f}, "
                          f"Terminated={terminated}, Truncated={truncated}")
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Episode Length: {step_count} steps")
            
    finally:
        # Always close the environment
        env.close()
    
    # Print evaluation summary
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Number of Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained SARSA agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the trained model (default: saved_models/sarsa_model_final.pth)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        render=not args.no_render,
        num_episodes=args.episodes
    )

import os
import time
import numpy as np
import torch

from config import Config
from environment.cartpole_env import CartPoleEnv
from agents.double_dqn_agent import DoubleDQNAgent


def evaluate(model_path=None, render=True, num_episodes=10):
    config = Config()
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_PATH, "dqn_model_final.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train.py")
        return
    render_mode = "human" if render else "rgb_array"
    env = CartPoleEnv(config.ENV_NAME, render_mode)
    agent = DoubleDQNAgent(env.input_dim, env.output_dim, config)
    agent.load_model(model_path)
    print(f"\nLoaded model from: {model_path}")
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
                    time.sleep(0.01)
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1
                state = next_state
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
        env.close()
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
    
    parser = argparse.ArgumentParser(description="Evaluate trained Double DQN agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the trained model (default: saved_models/dqn_model_final.pth)")
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

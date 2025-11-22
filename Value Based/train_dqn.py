import os
import numpy as np
from tqdm import tqdm

from config import Config
from environment.cartpole_env import CartPoleEnv
from agents.dqn_agent import DQNAgent
from utils.visualization import plot_rewards, plot_rolling_average, plot_training_summary


def train():
    config = Config()
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.PLOT_SAVE_PATH, exist_ok=True)
    env = CartPoleEnv(config.ENV_NAME, config.RENDER_MODE)
    print(f"Environment: {config.ENV_NAME}")
    print(f"State space: {env.input_dim}, Action space: {env.output_dim}")
    agent = DQNAgent(env.input_dim, env.output_dim, config)
    episode_rewards = []
    
    print(f"\nStarting training for {config.NUM_EPISODES} episodes...")
    print("-" * 60)
    for episode in tqdm(range(config.NUM_EPISODES), desc="Training Progress"):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            state = next_state
            step_count += 1
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        agent.update_target_network()
        if (episode + 1) % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-config.LOG_INTERVAL:])
            print(f"\nEpisode {episode + 1}/{config.NUM_EPISODES}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Average Reward (last {config.LOG_INTERVAL}): {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Steps: {step_count}")
        if (episode + 1) % config.SAVE_INTERVAL == 0:
            model_path = os.path.join(config.MODEL_SAVE_PATH, f"dqn_model_episode_{episode + 1}.pth")
            agent.save_model(model_path)
    env.close()
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, "dqn_model_final.pth")
    agent.save_model(final_model_path)
    print("\nGenerating training plots...")
    plot_rewards(episode_rewards, config.PLOT_SAVE_PATH)
    plot_rolling_average(episode_rewards, window=100, save_path=config.PLOT_SAVE_PATH)
    plot_training_summary(episode_rewards, config.PLOT_SAVE_PATH)
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

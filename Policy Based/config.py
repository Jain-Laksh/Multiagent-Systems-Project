"""
Configuration file for policy-based RL agents (Actor-Critic, REINFORCE, REINFORCE with Baseline) on CartPole-v1
Contains all hyperparameters and environment settings
Shared across all three implementations
"""

class Config:
    """Configuration class for Actor-Critic, REINFORCE, and REINFORCE with Baseline agents"""
    
    # Environment settings
    ENV_NAME = "CartPole-v1"
    RENDER_MODE = "rgb_array"  # Use 'human' for visualization during training
    
    # Network architecture
    INPUT_DIM = 4  # CartPole observation space dimension
    OUTPUT_DIM = 2  # CartPole action space dimension
    HIDDEN_DIM_1 = 128
    HIDDEN_DIM_2 = 128
    
    # Training hyperparameters
    ACTOR_LEARNING_RATE = 0.001  # Learning rate for policy (actor)
    CRITIC_LEARNING_RATE = 0.005  # Learning rate for value function (critic/baseline)
    GAMMA = 0.99  # Discount factor
    ENTROPY_COEF = 0.01  # Entropy coefficient for exploration
    
    # Training settings
    NUM_EPISODES = 200  # Match the reference script
    MAX_STEPS_PER_EPISODE = 500
    SEED = 42  # Random seed for reproducibility
    
    # Logging and saving
    LOG_INTERVAL = 50  # Print progress every N episodes
    SAVE_INTERVAL = 100  # Save model every N episodes
    MODEL_SAVE_PATH = "saved_models"
    PLOT_SAVE_PATH = "plots"
    
    # Evaluation settings
    EVAL_EPISODES = 100
    
    # Device settings
    DEVICE = "cuda"  # Use "cuda" if available, else "cpu"


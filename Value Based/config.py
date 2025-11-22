class Config:
    
    # Environment settings
    ENV_NAME = "CartPole-v1"
    RENDER_MODE = "rgb_array"  
    
    # Network architecture
    INPUT_DIM = 4  # CartPole observation space dimension
    OUTPUT_DIM = 2  # CartPole action space dimension
    HIDDEN_DIM_1 = 64
    HIDDEN_DIM_2 = 64
    
    # Training hyperparameters
    LEARNING_RATE = 0.0001
    GAMMA = 0.9  # Discount factor
    
    # Exploration parameters
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.0001
    
    # Replay buffer
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    
    # Training settings
    NUM_EPISODES = 1200
    TAU = 0.01  # Soft update parameter for target network
    
    # Logging and saving
    LOG_INTERVAL = 50  # Print progress every N episodes
    SAVE_INTERVAL = 100  # Save model every N episodes
    MODEL_SAVE_PATH = "saved_models"
    PLOT_SAVE_PATH = "plots"
    
    # Evaluation settings
    EVAL_EPISODES = 100
    MAX_STEPS_PER_EPISODE = 500
    
    # Device settings
    DEVICE = "cuda"  # Use "cuda" if available, else "cpu"

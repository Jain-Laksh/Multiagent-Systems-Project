"""
Environment wrapper for CartPole-v1
Handles environment creation and interaction
"""

import gymnasium as gym


class CartPoleEnv:
    """Wrapper class for CartPole-v1 environment"""
    
    def __init__(self, env_name="CartPole-v1", render_mode="rgb_array"):
        """
        Initialize the CartPole environment
        
        Args:
            env_name (str): Name of the Gymnasium environment
            render_mode (str): Render mode ('human', 'rgb_array', or None)
        """
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = gym.make(env_name, render_mode=render_mode)
        
        # Get environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.input_dim = self.observation_space.shape[0]
        self.output_dim = self.action_space.n
    
    def reset(self, seed=None):
        """
        Reset the environment
        
        Args:
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            tuple: Initial state and info dictionary
        """
        if seed is not None:
            state, info = self.env.reset(seed=seed)
        else:
            state, info = self.env.reset()
        return state, info
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): Action to take
            
        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
        """
        return self.env.step(action)
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def sample_action(self):
        """
        Sample a random action from the action space
        
        Returns:
            int: Random action
        """
        return self.action_space.sample()

"""
REINFORCE Agent implementation (Monte Carlo Policy Gradient)
Uses complete episode returns to update the policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from models.policy_network import PolicyNetwork


class REINFORCEAgent:
    """
    REINFORCE Agent (Vanilla Policy Gradient)
    Updates policy using Monte Carlo returns
    """
    
    def __init__(self, input_dim, output_dim, config):
        """
        Initialize the REINFORCE agent
        
        Args:
            input_dim (int): State space dimension
            output_dim (int): Action space dimension
            config: Configuration object with hyperparameters
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Set device
        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Initialize policy network
        self.policy = PolicyNetwork(
            input_dim,
            output_dim,
            config.HIDDEN_DIM_1,
            config.HIDDEN_DIM_2
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.ACTOR_LEARNING_RATE)
        
        # Hyperparameters
        self.gamma = config.GAMMA
        
        # Episode memory
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state, training=True):
        """
        Select an action based on the current policy
        
        Args:
            state: Current state
            training (bool): Whether in training mode (sample from distribution) or evaluation mode (greedy)
            
        Returns:
            int: Selected action
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        action_probs = self.policy(state_tensor)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        if training:
            # Sample action from distribution
            action = dist.sample()
            log_prob = dist.log_prob(action)
            self.log_probs.append(log_prob)
            return action.item()
        else:
            # Greedy action selection
            action = torch.argmax(action_probs)
            return action.item()
    
    def store_reward(self, reward):
        """
        Store reward for current step
        
        Args:
            reward: Reward received
        """
        self.rewards.append(reward)
    
    def train_step(self):
        """
        Perform one training step at the end of an episode
        Uses Monte Carlo returns with discounting
        
        Returns:
            float: Policy loss
        """
        # Calculate returns (discounted cumulative rewards)
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensor and normalize
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode memory
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def reset_episode(self):
        """
        Reset episode memory (log probs and rewards)
        """
        self.log_probs = []
        self.rewards = []
    
    def save_model(self, filepath):
        """
        Save the policy network
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load saved policy network
        
        Args:
            filepath (str): Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

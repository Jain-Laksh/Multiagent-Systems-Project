"""
REINFORCE with Baseline Agent implementation
Uses value function as baseline to reduce variance in policy gradient
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from models.policy_network import PolicyNetwork, ValueNetwork


class REINFORCEBaselineAgent:
    """
    REINFORCE Agent with Baseline (Value Function)
    Reduces variance in policy gradient updates by subtracting baseline
    """
    
    def __init__(self, input_dim, output_dim, config):
        """
        Initialize the REINFORCE with Baseline agent
        
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
        
        # Initialize value network (baseline)
        self.value = ValueNetwork(
            input_dim,
            config.HIDDEN_DIM_1,
            config.HIDDEN_DIM_2
        ).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.ACTOR_LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config.CRITIC_LEARNING_RATE)
        
        # Hyperparameters
        self.gamma = config.GAMMA
        
        # Episode memory
        self.log_probs = []
        self.rewards = []
        self.states = []
    
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
            self.states.append(state)
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
        Uses Monte Carlo returns with baseline to reduce variance
        
        Returns:
            tuple: (policy_loss, value_loss)
        """
        # Calculate returns (discounted cumulative rewards)
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensors
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        
        # Get baseline (value estimates)
        values = self.value(states_tensor).squeeze()
        
        # Calculate advantages (returns - baseline)
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate policy loss with baseline
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Calculate value loss (MSE between predicted values and returns)
        value_loss = nn.MSELoss()(values, returns)
        
        # Update value network (baseline)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Clear episode memory
        self.log_probs = []
        self.rewards = []
        self.states = []
        
        return policy_loss.item(), value_loss.item()
    
    def reset_episode(self):
        """
        Reset episode memory (log probs, rewards, and states)
        """
        self.log_probs = []
        self.rewards = []
        self.states = []
    
    def save_model(self, filepath):
        """
        Save the policy and value networks
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load saved policy and value networks
        
        Args:
            filepath (str): Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

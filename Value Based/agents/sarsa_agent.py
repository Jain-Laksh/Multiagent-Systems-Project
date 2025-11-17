"""
SARSA Agent implementation
On-policy TD control algorithm
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.dqn_network import DQNNetwork


class SARSAAgent:
    """
    SARSA Agent - On-policy Temporal Difference Learning
    State-Action-Reward-State-Action algorithm
    """
    
    def __init__(self, input_dim, output_dim, config):
        """
        Initialize the SARSA agent
        
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
        
        # Initialize Q-network (no target network needed for SARSA)
        self.q_network = DQNNetwork(
            input_dim, 
            output_dim,
            config.HIDDEN_DIM_1,
            config.HIDDEN_DIM_2
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        
        # Exploration parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN
        
        # Other parameters
        self.gamma = config.GAMMA
    
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            training (bool): Whether in training mode (use epsilon-greedy) or evaluation mode (greedy)
            
        Returns:
            int: Selected action
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randint(0, self.output_dim - 1)
        else:
            # Greedy action (exploitation)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def train_step(self, state, action, reward, next_state, next_action, done):
        """
        Perform one SARSA training step
        SARSA uses the actual next action taken (on-policy)
        
        Args:
            state: Current state
            action: Action taken in current state
            reward: Reward received
            next_state: Next state
            next_action: Actual action that will be taken in next state (on-policy)
            done: Whether the episode terminated
            
        Returns:
            float: Loss value
        """
        # Convert to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_action_tensor = torch.tensor([next_action], dtype=torch.long).to(self.device)
        done_tensor = torch.tensor([done], dtype=torch.float32).to(self.device)
        
        # Compute Q-value for current state-action pair
        q_values = self.q_network(state_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-value using SARSA update rule
        # Q(s,a) <- Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        # where a' is the actual next action taken (on-policy)
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            next_q_value = next_q_values.gather(1, next_action_tensor.unsqueeze(1)).squeeze(1)
            target_q_value = reward_tensor + (self.gamma * next_q_value * (1 - done_tensor))
        
        # Compute loss
        loss = self.loss_fn(q_value, target_q_value)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """
        Decay epsilon for exploration
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """
        Save the Q-network model
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved Q-network model
        
        Args:
            filepath (str): Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

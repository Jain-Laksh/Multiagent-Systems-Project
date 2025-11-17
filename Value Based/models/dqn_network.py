"""
Deep Q-Network (DQN) architecture for Double DQN
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with fully connected layers
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim_1=64, hidden_dim_2=64):
        """
        Initialize the DQN network
        
        Args:
            input_dim (int): Dimension of input (state space)
            output_dim (int): Dimension of output (action space)
            hidden_dim_1 (int): Number of neurons in first hidden layer
            hidden_dim_2 (int): Number of neurons in second hidden layer
        """
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.network(x)

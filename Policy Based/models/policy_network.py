"""
Policy Network for REINFORCE algorithms
Outputs action probabilities for discrete action spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Policy Network that outputs action probabilities
    Used for REINFORCE and REINFORCE with Baseline
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim_1=128, hidden_dim_2=128):
        """
        Initialize the policy network
        
        Args:
            input_dim (int): Dimension of input (state space)
            output_dim (int): Dimension of output (action space)
            hidden_dim_1 (int): Number of neurons in first hidden layer (hidden_dim_2 not used, kept for compatibility)
        """
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, output_dim)
        self.dropout = nn.Dropout(p=0.6)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Action probabilities (after softmax)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """
    Value Network that outputs state value
    Used as baseline in REINFORCE with Baseline
    """
    
    def __init__(self, input_dim, hidden_dim_1=128, hidden_dim_2=128):
        """
        Initialize the value network
        
        Args:
            input_dim (int): Dimension of input (state space)
            hidden_dim_1 (int): Number of neurons in first hidden layer (hidden_dim_2 not used, kept for compatibility)
        """
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, 1)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: State value estimate
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)

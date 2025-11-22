"""
Actor-Critic Network Architecture
Implements both actor (policy) and critic (value function) networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    Actor Network that outputs action probabilities
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim_1=128, hidden_dim_2=128):
        """
        Initialize the actor network
        
        Args:
            input_dim (int): Dimension of input (state space)
            output_dim (int): Dimension of output (action space)
            hidden_dim_1 (int): Number of neurons in first hidden layer (hidden_dim_2 not used, kept for compatibility)
        """
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, output_dim)
        self.dropout = nn.Dropout(p=0.6)
    
    def forward(self, x):
        """
        Forward pass through the actor network
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Action probabilities (after softmax)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    """
    Critic Network that outputs state value estimate
    """
    
    def __init__(self, input_dim, hidden_dim_1=128, hidden_dim_2=128):
        """
        Initialize the critic network
        
        Args:
            input_dim (int): Dimension of input (state space)
            hidden_dim_1 (int): Number of neurons in first hidden layer (hidden_dim_2 not used, kept for compatibility)
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, 1)
    
    def forward(self, x):
        """
        Forward pass through the critic network
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: State value estimate
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)

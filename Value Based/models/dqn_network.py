import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1=64, hidden_dim_2=64):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

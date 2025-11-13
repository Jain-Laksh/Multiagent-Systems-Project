import torch.nn as nn


class PolicyNet(nn.Module):
    
    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

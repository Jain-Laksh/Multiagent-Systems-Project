import torch.nn as nn


class ActorCriticNet(nn.Module):
    
    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.net(x)
        action_logits = self.policy_head(features)
        state_value = self.value_head(features).squeeze(-1)
        return action_logits, state_value

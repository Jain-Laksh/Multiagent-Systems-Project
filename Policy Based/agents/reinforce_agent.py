import torch
import torch.optim as optim

from models.policy import PolicyNet
from utils.helpers import one_hot


class REINFORCEAgent:
    
    def __init__(self, env, lr=1e-3, gamma=0.99, hidden=128, device='cpu'):
        self.env = env
        self.gamma = gamma
        self.device = device
        
        n_states = env.n_states
        n_actions = env.n_actions
        
        self.policy = PolicyNet(n_states, n_actions, hidden).to(device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.tensor(one_hot(self.env.n_states, state)).unsqueeze(0).to(self.device)
        logits = self.policy(state_tensor)
        action_probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        
        return action.item(), distribution.log_prob(action)

    def train_episode(self):
        state = self.env.reset()
        trajectory = []
        done = False
        episode_reward = 0.0
        
        while not done:
            action, log_prob = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            trajectory.append((state, action, reward, log_prob))
            episode_reward += reward
            state = next_state

        monte_carlo_returns = []
        cumulative_return = 0.0
        for _, _, reward, _ in reversed(trajectory):
            cumulative_return = reward + self.gamma * cumulative_return
            monte_carlo_returns.insert(0, cumulative_return)
        
        returns_tensor = torch.tensor(monte_carlo_returns, dtype=torch.float32).to(self.device)

        policy_gradient_loss = 0.0
        for (_, _, _, log_prob), return_value in zip(trajectory, returns_tensor):
            policy_gradient_loss = policy_gradient_loss - log_prob * return_value

        self.opt.zero_grad()
        policy_gradient_loss.backward()
        self.opt.step()

        return episode_reward

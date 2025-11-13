import torch
import torch.optim as optim

from models.actor_critic import ActorCriticNet
from utils.helpers import one_hot


class ActorCriticAgent:
    
    def __init__(self, env, lr=1e-3, gamma=0.99, hidden=128, device='cpu'):
        self.env = env
        self.gamma = gamma
        self.device = device
        
        n_states = env.n_states
        n_actions = env.n_actions
        
        self.model = ActorCriticNet(n_states, n_actions, hidden).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.tensor(one_hot(self.env.n_states, state)).unsqueeze(0).to(self.device)
        logits, value = self.model(state_tensor)
        action_probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        
        return action.item(), distribution.log_prob(action), value.squeeze(0)

    def train_episode(self, max_steps=None):
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action, log_prob, value = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            state_tensor = torch.tensor(one_hot(self.env.n_states, state)).unsqueeze(0).to(self.device)
            next_state_tensor = torch.tensor(one_hot(self.env.n_states, next_state)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, next_value = self.model(next_state_tensor)
            
            td_target = reward + (0.0 if done else self.gamma * next_value.item())
            advantage = td_target - value.item()

            policy_loss = -log_prob * advantage
            value_loss = 0.5 * (td_target - value) ** 2

            total_loss = policy_loss + value_loss
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

            state = next_state
            steps += 1
            
            if max_steps and steps >= max_steps:
                break

        return episode_reward

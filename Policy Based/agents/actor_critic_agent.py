import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from models.actor_critic_network import ActorNetwork, CriticNetwork


class ActorCriticAgent:
    """
    Actor-Critic Agent with separate actor and critic networks
    """

    def __init__(self, input_dim, output_dim, config):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        # Device
        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
        )
        print(f"Using device: {self.device}")

        # Networks
        self.actor = ActorNetwork(
            input_dim,
            output_dim,
            config.HIDDEN_DIM_1,
            config.HIDDEN_DIM_2
        ).to(self.device)

        self.critic = CriticNetwork(
            input_dim,
            config.HIDDEN_DIM_1,
            config.HIDDEN_DIM_2
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LEARNING_RATE)

        self.gamma = config.GAMMA
        self.entropy_coef = getattr(config, "ENTROPY_COEF", 0.0)

    def select_action(self, state, training=True):
      state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
      
      # Always compute with grad disabled here
      with torch.no_grad():
          action_probs = self.actor(state_tensor)
      
      dist = Categorical(action_probs)

      if training:
          action = dist.sample()
          log_prob = dist.log_prob(action)
          entropy = dist.entropy().mean()
          return action.item(), log_prob, entropy
      else:
          action = torch.argmax(action_probs)
          return action.item()


    def train_step(self, state, action, reward, next_state, done, log_prob=None, entropy=None):        
      """
        One Actor-Critic update.
        We recompute log_prob and entropy from current policy.
        """

        # ----- Tensors -----
      state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
      next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
      reward_tensor = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
      done_tensor = torch.as_tensor([done], dtype=torch.float32, device=self.device)

      # ----- Critic update -----
      value = self.critic(state_tensor)               # V(s_t)
      next_value = self.critic(next_state_tensor)     # V(s_{t+1})

      with torch.no_grad():
          td_target = reward_tensor + self.gamma * next_value * (1 - done_tensor)

      td_error = td_target - value                    # advantage estimate (1-step)

      critic_loss = td_error.pow(2).mean()

      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      # ----- Actor update -----
      # Recompute distribution for current state with grad
      action_probs = self.actor(state_tensor)
      dist = Categorical(action_probs)

      action_tensor = torch.as_tensor([action], dtype=torch.int64, device=self.device)
      log_prob_current = dist.log_prob(action_tensor)

      entropy = dist.entropy().mean()
      advantage = td_error.detach()                   # stop grad into critic

      actor_loss = -(log_prob_current * advantage + self.entropy_coef * entropy)

      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      return actor_loss.item(), critic_loss.item()

    def save_model(self, filepath):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print(f"Model loaded from {filepath}")

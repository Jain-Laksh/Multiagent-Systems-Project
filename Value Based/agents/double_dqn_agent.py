import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.dqn_network import DQNNetwork
from utils.replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    def __init__(self, input_dim, output_dim, config):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
        )
        print(f"Using device: {self.device}")
        self.q_network = DQNNetwork(
            input_dim, 
            output_dim,
            config.HIDDEN_DIM_1,
            config.HIDDEN_DIM_2
        ).to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        q_values = self.q_network(states_tensor)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            next_q_value = next_q_values.max(dim=1)[0]
            target_q_value = rewards_tensor + (self.gamma * next_q_value * (1 - dones_tensor))
        
        loss = self.loss_fn(q_value, target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        for target_param, q_param in zip(self.target_network.parameters(), 
                                         self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * q_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

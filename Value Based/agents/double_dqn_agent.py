"""
Double DQN Agent implementation
Handles action selection, training, and model updates
"""

import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.dqn_network import DQNNetwork
from utils.replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """
    Double DQN Agent with experience replay and target network
    """
    
    def __init__(self, input_dim, output_dim, config):
        """
        Initialize the Double DQN agent
        
        Args:
            input_dim (int): State space dimension
            output_dim (int): Action space dimension
            config: Configuration object with hyperparameters
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Set device
        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Initialize Q-network and target network
        self.q_network = DQNNetwork(
            input_dim, 
            output_dim,
            config.HIDDEN_DIM_1,
            config.HIDDEN_DIM_2
        ).to(self.device)
        
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()  # Set to evaluation mode
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
        
        # Exploration parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN
        
        # Other parameters
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
    
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            training (bool): Whether in training mode (use epsilon-greedy) or evaluation mode (greedy)
            
        Returns:
            int: Selected action
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randint(0, self.output_dim - 1)
        else:
            # Greedy action (exploitation)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step (sample batch and update Q-network)
        
        Returns:
            float: Loss value, or None if buffer is not ready
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample a batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute Q-values from Q-network
        q_values = self.q_network(states_tensor)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network (Double DQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            next_q_value = next_q_values.max(dim=1)[0]
            target_q_value = rewards_tensor + (self.gamma * next_q_value * (1 - dones_tensor))
        
        # Compute loss
        loss = self.loss_fn(q_value, target_q_value)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Soft update of target network parameters
        θ_target = τ * θ_q + (1 - τ) * θ_target
        """
        for target_param, q_param in zip(self.target_network.parameters(), 
                                         self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * q_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def decay_epsilon(self):
        """
        Decay epsilon for exploration
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """
        Save the Q-network model
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved Q-network model
        
        Args:
            filepath (str): Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

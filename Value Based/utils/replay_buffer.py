"""
Experience Replay Buffer for storing and sampling transitions
"""

import random
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    
    def __init__(self, buffer_size=10000):
        """
        Initialize the replay buffer
        
        Args:
            buffer_size (int): Maximum size of the buffer
        """
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            list: List of sampled transitions
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """
        Get the current size of the buffer
        
        Returns:
            int: Current buffer size
        """
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """
        Check if the buffer has enough samples for training
        
        Args:
            batch_size (int): Required batch size
            
        Returns:
            bool: True if buffer has enough samples
        """
        return len(self.buffer) >= batch_size

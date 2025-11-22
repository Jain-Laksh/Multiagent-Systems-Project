import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Hyperparameters ---
LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.005  # Critics often need higher LR to track values fast
GAMMA = 0.99
EPISODES = 200
HIDDEN_SIZE = 128
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Neural Network Models ---

class PolicyNetwork(nn.Module):
    """The Actor: Outputs action probabilities."""
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, output_dim)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class ValueNetwork(nn.Module):
    """The Critic/Baseline: Outputs a single scalar value V(s)."""
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- Helper Functions ---

def calculate_returns(rewards, gamma):
    """Calculates discounted returns (G_t) for an entire episode."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    return returns

def smooth_curve(data, window=50):
    """Calculates rolling mean for smoother plotting."""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

# --- 1. Vanilla REINFORCE Agent ---

def train_reinforce(env_name, episodes):
    print(f"Training REINFORCE on {env_name}...")
    env = gym.make(env_name)
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE_ACTOR)
    
    scores = []

    for ep in range(episodes):
        state, _ = env.reset(seed=SEED+ep)
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state_tensor)
            
            # Sample action
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            log_probs.append(m.log_prob(action))
            
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)

        scores.append(sum(rewards))
        
        # --- REINFORCE Update ---
        returns = calculate_returns(rewards, GAMMA)
        
        # Normalize returns (Crucial for Vanilla REINFORCE stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        loss = []
        for log_prob, Gt in zip(log_probs, returns):
            loss.append(-log_prob * Gt)
            
        optimizer.zero_grad()
        policy_loss = torch.stack(loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}, Score: {np.mean(scores[-50:]):.2f}")

    env.close()
    return scores

# --- 2. REINFORCE with Baseline Agent ---

def train_reinforce_baseline(env_name, episodes):
    print(f"\nTraining REINFORCE with Baseline on {env_name}...")
    env = gym.make(env_name)
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    value_net = ValueNetwork(env.observation_space.shape[0])
    
    optimizer_policy = optim.Adam(policy.parameters(), lr=LEARNING_RATE_ACTOR)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE_CRITIC)
    
    scores = []

    for ep in range(episodes):
        state, _ = env.reset(seed=SEED+ep)
        log_probs = []
        rewards = []
        states = []
        done = False
        
        while not done:
            states.append(state)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))
            
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)

        scores.append(sum(rewards))
        
        # --- Update Step ---
        returns = calculate_returns(rewards, GAMMA)
        
        # Get value estimates for all visited states
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        values = value_net(states_tensor).squeeze()
        
        # Calculate Advantage: G_t - V(s_t)
        # Detach values from graph because we don't want to update Value net via Actor loss
        advantages = returns - values.detach()
        
        # 1. Update Actor
        policy_loss = []
        for log_prob, adv in zip(log_probs, advantages):
            policy_loss.append(-log_prob * adv)
        policy_loss = torch.stack(policy_loss).sum()
        
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        
        # 2. Update Critic (Baseline) - MSE between actual Return and Predicted Value
        value_loss = F.mse_loss(values, returns)
        
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}, Score: {np.mean(scores[-50:]):.2f}")

    env.close()
    return scores

# --- 3. Actor-Critic (Online) Agent ---

def train_actor_critic(env_name, episodes):
    print(f"\nTraining Actor-Critic (Online) on {env_name}...")
    env = gym.make(env_name)
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    value_net = ValueNetwork(env.observation_space.shape[0])
    
    optimizer_policy = optim.Adam(policy.parameters(), lr=LEARNING_RATE_ACTOR)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE_CRITIC)
    
    scores = []

    for ep in range(episodes):
        state, _ = env.reset(seed=SEED+ep)
        done = False
        score = 0
        I = 1  # Discount factor accumulator
        
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            # Get Action
            probs = policy(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            
            # Take Step
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            score += reward
            
            # Prepare Next State
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
            
            # Calculate TD Target
            # If terminal, value of next state is 0
            v_state = value_net(state_tensor)
            v_next_state = value_net(next_state_tensor) if not done else torch.tensor([[0.0]])
            
            td_target = reward + GAMMA * v_next_state.item() * (1 - int(done))
            td_error = td_target - v_state
            
            # 1. Critic Update (MSE)
            # We detach target to keep it fixed like a label
            critic_loss = F.mse_loss(v_state, torch.tensor([[td_target]]))
            
            optimizer_value.zero_grad()
            critic_loss.backward()
            optimizer_value.step()
            
            # 2. Actor Update
            # Update using TD error as the advantage estimate
            # We detach td_error because actor shouldn't update critic weights
            actor_loss = -log_prob * td_error.detach() * I
            
            optimizer_policy.zero_grad()
            actor_loss.backward()
            optimizer_policy.step()
            
            state = next_state
            I *= GAMMA # Decay importance of future updates if using I (optional for simple AC but good practice)

        scores.append(score)
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}, Score: {np.mean(scores[-50:]):.2f}")

    env.close()
    return scores

# --- Main Execution & Plotting ---

if __name__ == "__main__":
    env_name = "CartPole-v1"
    
    # Train all three agents
    scores_reinforce = train_reinforce(env_name, EPISODES)
    scores_reinforce_baseline = train_reinforce_baseline(env_name, EPISODES)
    scores_ac = train_actor_critic(env_name, EPISODES)

    # Smoothing
    smooth_reinforce = smooth_curve(scores_reinforce)
    smooth_reinforce_baseline = smooth_curve(scores_reinforce_baseline)
    smooth_ac = smooth_curve(scores_ac)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(smooth_ac, label='Actor-Critic', color='blue', linewidth=2)
    plt.plot(smooth_reinforce_baseline, label='REINFORCE with Baseline', color='green', linewidth=2)
    plt.plot(smooth_reinforce, label='REINFORCE', color='red', linewidth=2, alpha=0.7)
    
    plt.title(f'Actor-Critic vs REINFORCE vs Baseline: {env_name} Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Reward (50 episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('cartpole_comparison.png')
    print("Plot saved to 'cartpole_comparison.png'")
    plt.show()

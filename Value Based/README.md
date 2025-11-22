# Value-Based Reinforcement Learning Implementations

Complete implementations of **DQN**, **Double DQN**, and **SARSA** algorithms for solving the CartPole-v1 environment using Gymnasium.

## Overview

This project implements three value-based reinforcement learning algorithms:
- **DQN (Deep Q-Network)** - Off-policy TD learning with experience replay
- **Double DQN** - Reduces Q-value overestimation bias
- **SARSA** - On-policy TD learning algorithm

## Project Structure

```
Value Based/
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py             # DQN agent implementation
│   ├── double_dqn_agent.py      # Double DQN agent implementation
│   └── sarsa_agent.py           # SARSA agent implementation
├── environment/
│   ├── __init__.py
│   └── cartpole_env.py          # CartPole environment wrapper
├── models/
│   ├── __init__.py
│   └── dqn_network.py           # Neural network architecture
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py         # Experience replay buffer (for DQN/Double DQN)
│   └── visualization.py         # Plotting utilities
├── saved_models/                # Trained models (created during training)
├── plots/                       # Training plots (created during training)
├── config.py                    # Configuration and hyperparameters
├── train_dqn.py                 # Training script for DQN
├── train_double_dqn.py          # Training script for Double DQN
├── train_sarsa.py               # Training script for SARSA
├── evaluate_dqn.py              # Evaluation script for DQN
├── evaluate_double_dqn.py       # Evaluation script for Double DQN
├── evaluate_sarsa.py            # Evaluation script for SARSA
├── compare_all_agents.py        # Compare all three algorithms
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── QUICK_START.md               # Quick start guide
```

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `torch` - Deep learning framework
- `gymnasium` - RL environments
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `pandas` - Data analysis
- `tqdm` - Progress bars

## Quick Start

See [QUICK_START.md](QUICK_START.md) for detailed instructions.

### Training

Train any of the three agents:

```bash
# Train DQN
python train_dqn.py

# Train Double DQN
python train_double_dqn.py

# Train SARSA
python train_sarsa.py
```

### Evaluation

Evaluate a trained agent:

```bash
# Evaluate DQN
python evaluate_dqn.py

# Evaluate with custom options
python evaluate_dqn.py --model saved_models/dqn_model_episode_1000.pth --episodes 20 --no-render
```

## Algorithm Comparison

### DQN (Deep Q-Network) - Off-Policy
- **Learning**: Uses experience replay to learn from past experiences
- **Target Network**: Uses target network for both action selection and evaluation
- **Update Rule**: `Q_target = r + γ * max_a' Q_target(s', a')`
- **Advantages**:
  - Sample efficient through experience replay
  - Can reuse past experiences
  - Stable learning with target network
- **Disadvantages**:
  - Can overestimate Q-values
  - Max operator introduces positive bias

### Double DQN - Off-Policy
- **Learning**: Uses experience replay like DQN
- **Target Network**: Decouples action selection (online network) from evaluation (target network)
- **Update Rule**: `Q_target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))`
- **Advantages**:
  - Reduces overestimation bias
  - More stable learning than DQN
  - Better performance on complex tasks
- **Disadvantages**:
  - Slightly more complex implementation
  - Similar computational cost to DQN

### SARSA (State-Action-Reward-State-Action) - On-Policy
- **Learning**: Learns from sequential experience without replay buffer
- **Target**: Uses actual next action from current policy
- **Update Rule**: `Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]`
- **Advantages**:
  - Accounts for exploration in learning
  - More conservative and safer
  - Simpler implementation (no replay buffer)
- **Disadvantages**:
  - Less sample efficient
  - Can be slower to converge
  - Cannot reuse past experiences

### Key Differences

| Feature | DQN | Double DQN | SARSA |
|---------|-----|------------|-------|
| Policy | Off-policy | Off-policy | On-policy |
| Experience Replay | Yes | Yes | No |
| Target Network | Yes | Yes | No |
| Q-value Bias | Overestimation | Reduced | Minimal |
| Sample Efficiency | High | High | Lower |
| Convergence | Fast | Fast | Moderate |
| Exploration Handling | Separate | Separate | Integrated |

## Configuration

All hyperparameters are in `config.py`:

### Network Architecture
```python
HIDDEN_DIM_1 = 64  # First hidden layer size
HIDDEN_DIM_2 = 64  # Second hidden layer size
```

### Training Parameters
```python
LEARNING_RATE = 0.0001
GAMMA = 0.9           # Discount factor
NUM_EPISODES = 1200
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TAU = 0.01            # Soft update parameter for target network
```

### Exploration Parameters
```python
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.0001
```

### Customization Examples

**Train for more episodes:**
```python
NUM_EPISODES = 2000
EPSILON_DECAY = 0.997  # Slower decay for longer training
```

**Use larger network:**
```python
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 128
```

**Different environment:**
```python
ENV_NAME = "LunarLander-v2"
INPUT_DIM = 8
OUTPUT_DIM = 4
```

## File Descriptions

### Core Files
- **config.py**: All hyperparameters and configuration settings
- **train_*.py**: Training scripts for each algorithm
- **evaluate_*.py**: Evaluation scripts for testing trained agents
- **compare_all_agents.py**: Comprehensive comparison tool

### Agent Implementations
- **agents/dqn_agent.py**: Standard DQN with experience replay and target network
- **agents/double_dqn_agent.py**: Double DQN with action decoupling
- **agents/sarsa_agent.py**: On-policy SARSA without experience replay

### Supporting Modules
- **models/dqn_network.py**: Neural network architecture (3-layer MLP)
- **environment/cartpole_env.py**: Environment wrapper for CartPole
- **utils/replay_buffer.py**: Experience replay buffer (used by DQN/Double DQN)
- **utils/visualization.py**: Plotting and visualization utilities

## License

This project is for educational purposes as part of the FAI course.

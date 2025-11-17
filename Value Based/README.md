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

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Three Algorithms**: DQN, Double DQN, and SARSA implementations
- **Experience Replay**: Efficient replay buffer for off-policy methods
- **Soft Target Updates**: Smooth target network updates (DQN/Double DQN)
- **Epsilon-Greedy Exploration**: Decaying epsilon for exploration-exploitation tradeoff
- **Comprehensive Logging**: Progress tracking and model checkpointing
- **Visualization**: Automated generation of training plots and metrics
- **GPU Support**: Automatic detection and usage of CUDA if available
- **Comparison Tools**: Compare performance across all algorithms

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

### Comparison

Compare all three algorithms:

```bash
python compare_all_agents.py
```

## Algorithm Comparison

### DQN (Off-Policy)
- **Learning**: Uses experience replay to learn from past experiences
- **Target**: Uses target network for both action selection and evaluation
- **Update**: `Q_target = r + γ * max_a' Q_target(s', a')`
- **Pros**: Sample efficient, can reuse experiences
- **Cons**: Can overestimate Q-values

### Double DQN (Off-Policy)
- **Learning**: Uses experience replay like DQN
- **Target**: Decouples action selection (online network) from evaluation (target network)
- **Update**: `Q_target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))`
- **Pros**: Reduces overestimation bias, more stable learning
- **Cons**: Slightly more complex than DQN

### SARSA (On-Policy)
- **Learning**: Learns from sequential experience (no replay buffer)
- **Target**: Uses actual next action from current policy
- **Update**: `Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]`
- **Pros**: Accounts for exploration in learning, more conservative
- **Cons**: Less sample efficient, can be slower to converge

## Configuration

All hyperparameters are in `config.py`:

```python
# Network architecture
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 64

# Training hyperparameters
LEARNING_RATE = 0.0001
GAMMA = 0.9  # Discount factor

# Exploration
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.0001

# Training settings
NUM_EPISODES = 1200
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TAU = 0.01  # Soft update parameter
```

## Expected Performance

On CartPole-v1:
- **DQN**: ~450-500 average reward after 1000-1200 episodes
- **Double DQN**: Similar or slightly better with more stable learning
- **SARSA**: ~450-500 average reward, may be more conservative

All agents should solve the environment (achieve 475+ average reward consistently).

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

## Training Output

During training, you'll see:
- Progress bars with episode completion
- Periodic logging of rewards and epsilon values
- Model checkpoints saved every 100 episodes
- Final model saved at completion
- Training plots generated automatically

## Evaluation Output

When evaluating, you'll get:
- Episode-by-episode performance
- Visual rendering of agent behavior (optional)
- Statistics: mean, std, max, min rewards
- Episode length statistics

## References

1. **DQN**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
2. **Double DQN**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015)
3. **SARSA**: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (Sutton & Barto, 2018)
4. **Gymnasium**: https://gymnasium.farama.org/

## License

This project is for educational purposes as part of the FAI course.

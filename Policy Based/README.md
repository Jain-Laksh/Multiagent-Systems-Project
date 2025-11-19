# Policy-Based Reinforcement Learning Implementations

Complete implementations of **Actor-Critic**, **REINFORCE**, and **REINFORCE with Baseline** algorithms for solving the CartPole-v1 environment using Gymnasium.

## Overview

This project implements three policy-based reinforcement learning algorithms:
- **Actor-Critic** - Combines policy gradient (actor) with value function estimation (critic) for step-by-step updates
- **REINFORCE** - Vanilla Monte Carlo policy gradient algorithm
- **REINFORCE with Baseline** - REINFORCE enhanced with value function baseline to reduce variance

## Project Structure

```
Policy Based/
├── agents/
│   ├── __init__.py
│   ├── actor_critic_agent.py          # Actor-Critic agent implementation
│   ├── reinforce_agent.py             # REINFORCE agent implementation
│   └── reinforce_baseline_agent.py    # REINFORCE with Baseline implementation
├── environment/
│   ├── __init__.py
│   └── cartpole_env.py                # CartPole environment wrapper
├── models/
│   ├── __init__.py
│   ├── actor_critic_network.py        # Actor and Critic network architectures
│   └── policy_network.py              # Policy and Value network architectures
├── utils/
│   ├── __init__.py
│   └── visualization.py               # Plotting utilities
├── saved_models/                      # Trained models (created during training)
├── plots/                             # Training plots (created during training)
├── config.py                          # Configuration and hyperparameters
├── train_actor_critic.py              # Training script for Actor-Critic
├── train_reinforce.py                 # Training script for REINFORCE
├── train_reinforce_baseline.py        # Training script for REINFORCE with Baseline
├── evaluate_actor_critic.py           # Evaluation script for Actor-Critic
├── evaluate_reinforce.py              # Evaluation script for REINFORCE
├── evaluate_reinforce_baseline.py     # Evaluation script for REINFORCE with Baseline
├── compare_all_agents.py              # Compare all three algorithms
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── QUICK_START.md                     # Quick start guide
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Three Policy-Gradient Algorithms**: Actor-Critic, REINFORCE, and REINFORCE with Baseline
- **Policy Gradient Methods**: Direct policy optimization without Q-values
- **Variance Reduction**: Baseline and critic networks to reduce gradient variance
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
# Train Actor-Critic
python train_actor_critic.py

# Train REINFORCE
python train_reinforce.py

# Train REINFORCE with Baseline
python train_reinforce_baseline.py
```

### Evaluation

Evaluate a trained agent:

```bash
# Evaluate Actor-Critic
python evaluate_actor_critic.py

# Evaluate with custom options
python evaluate_actor_critic.py --model saved_models/actor_critic_model_episode_1000.pth --episodes 20 --no-render
```

### Comparison

Compare all three algorithms:

```bash
python compare_all_agents.py
```

## Algorithm Comparison

### Actor-Critic
- **Learning**: Updates policy and value function at every step using TD error
- **Update**: Policy gradient scaled by TD error (advantage)
- **Pros**: Low variance, can learn from partial trajectories, fast updates
- **Cons**: Can be sensitive to hyperparameters

### REINFORCE (Vanilla Policy Gradient)
- **Learning**: Updates policy at end of episode using Monte Carlo returns
- **Update**: Policy gradient scaled by cumulative discounted rewards
- **Pros**: Simple, unbiased gradient estimates
- **Cons**: High variance, sample inefficient, slower convergence

### REINFORCE with Baseline
- **Learning**: Updates policy and baseline at end of episode
- **Update**: Policy gradient scaled by (returns - baseline) to reduce variance
- **Pros**: Reduced variance compared to vanilla REINFORCE, faster learning
- **Cons**: Still requires full episodes, more complex than vanilla REINFORCE

## Configuration

All hyperparameters are in `config.py`:

```python
# Network architecture
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 128

# Training hyperparameters
ACTOR_LEARNING_RATE = 0.0003  # Policy learning rate
CRITIC_LEARNING_RATE = 0.001  # Value function learning rate
GAMMA = 0.99  # Discount factor

# Training settings
NUM_EPISODES = 1200
MAX_STEPS_PER_EPISODE = 500
```

## Expected Performance

On CartPole-v1:
- **Actor-Critic**: ~450-500 average reward after 800-1000 episodes (fastest convergence)
- **REINFORCE**: ~450-500 average reward after 1000-1200 episodes (slower, high variance)
- **REINFORCE with Baseline**: ~450-500 average reward after 900-1100 episodes (better than vanilla REINFORCE)

All agents should solve the environment (achieve 475+ average reward consistently).

## File Descriptions

### Core Files
- **config.py**: All hyperparameters and configuration settings
- **train_*.py**: Training scripts for each algorithm
- **evaluate_*.py**: Evaluation scripts for testing trained agents
- **compare_all_agents.py**: Comprehensive comparison tool

### Agent Implementations
- **agents/actor_critic_agent.py**: Actor-Critic with separate actor and critic networks
- **agents/reinforce_agent.py**: Vanilla REINFORCE (Monte Carlo policy gradient)
- **agents/reinforce_baseline_agent.py**: REINFORCE with value function baseline

### Supporting Modules
- **models/actor_critic_network.py**: Actor and Critic network architectures
- **models/policy_network.py**: Policy and Value network architectures
- **environment/cartpole_env.py**: Environment wrapper for CartPole
- **utils/visualization.py**: Plotting and visualization utilities

## Key Differences from Value-Based Methods

1. **Direct Policy Learning**: Policy-based methods directly learn the policy (action probabilities) instead of learning Q-values
2. **Stochastic Policies**: Can naturally handle stochastic policies and continuous action spaces
3. **No Replay Buffer**: Most policy gradient methods (except Actor-Critic variants) don't use experience replay
4. **Episode-Based Updates**: REINFORCE methods update after complete episodes, not individual steps
5. **Gradient Variance**: Policy gradients can have high variance, requiring variance reduction techniques

## Training Output

During training, you'll see:
- Progress bars with episode completion
- Periodic logging of rewards
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

1. **Actor-Critic**: [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) (Sutton et al., 1999)
2. **REINFORCE**: [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) (Williams, 1992)
3. **Policy Gradients**: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (Sutton & Barto, 2018)
4. **Gymnasium**: https://gymnasium.farama.org/

## License

This project is for educational purposes as part of the FAI course.

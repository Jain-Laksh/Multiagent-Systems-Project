# Actor-Critic vs REINFORCE

A comparative implementation of Actor-Critic and REINFORCE reinforcement learning algorithms on a GridWorld environment.

## Project Structure

```
Actor_Critic/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── main.py               # Main entry point for training
├── config.py             # Configuration settings
├── environment/          # Environment implementations
│   ├── __init__.py
│   └── gridworld.py     # GridWorld environment
├── models/              # Neural network architectures
│   ├── __init__.py
│   ├── actor_critic.py  # Actor-Critic network
│   └── policy.py        # Policy network for REINFORCE
├── agents/              # RL Agent implementations
│   ├── __init__.py
│   ├── actor_critic_agent.py
│   └── reinforce_agent.py
└── utils/               # Utility functions
    ├── __init__.py
    ├── helpers.py       # Helper functions
    └── visualization.py # Plotting and visualization
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the default experiment:
```bash
python main.py
```

## Features

- **GridWorld Environment**: Customizable grid-based environment with goals and traps
- **Actor-Critic Algorithm**: Policy gradient with value function baseline
- **REINFORCE Algorithm**: Monte Carlo policy gradient
- **Visualization**: Training curves and policy demonstrations

## Algorithms

### Actor-Critic
Combines policy gradient with a value function critic that provides a baseline for variance reduction. Updates are performed at every step using TD error.

### REINFORCE
Monte Carlo policy gradient algorithm that updates based on complete episode returns.

## Results

The project compares both algorithms on a 6x6 GridWorld with traps, demonstrating the variance reduction benefits of Actor-Critic.

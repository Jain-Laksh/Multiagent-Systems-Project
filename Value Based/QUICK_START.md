# Quick Start Guide

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Train

```bash
# DQN
python train_dqn.py

# Double DQN
python train_double_dqn.py

# SARSA
python train_sarsa.py

# Compare all
python compare_all_agents.py
```

## Evaluate

```bash
# Evaluate trained agent
python evaluate_dqn.py
python evaluate_double_dqn.py
python evaluate_sarsa.py

# Custom evaluation
python evaluate_dqn.py --model saved_models/dqn_model_episode_1000.pth --episodes 20
```

For detailed information, see [README.md](README.md)

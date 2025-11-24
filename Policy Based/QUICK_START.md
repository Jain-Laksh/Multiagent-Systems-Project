# Quick Start Guide - Policy-Based RL Agents

This guide will help you quickly get started with training and evaluating Actor-Critic, REINFORCE, and REINFORCE with Baseline agents on CartPole-v1.

## Prerequisites

1. **Python 3.8+** installed
2. **Virtual environment** (recommended)

## Installation

### 1. Create and activate a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning)
- Gymnasium (RL environments)
- NumPy, Pandas (numerical computing)
- Matplotlib (visualization)
- tqdm (progress bars)

## Training Agents

### Train Actor-Critic

```bash
python train_actor_critic.py
```

**What happens:**
- Trains for 1200 episodes (configurable in `config.py`)
- Saves model checkpoints every 100 episodes
- Generates training plots automatically
- Final model saved as `saved_models/actor_critic_model_final.pth`

**Expected time:** ~10-20 minutes (depending on hardware)

### Train REINFORCE

```bash
python train_reinforce.py
```

**Output:**
- Model: `saved_models/reinforce_model_final.pth`
- Plots: `plots/reinforce_*.png`

### Train REINFORCE with Baseline

```bash
python train_reinforce_baseline.py
```

**Output:**
- Model: `saved_models/reinforce_baseline_model_final.pth`
- Plots: `plots/reinforce_baseline_*.png`

## Evaluating Agents

### Evaluate Actor-Critic

```bash
# With rendering (visual)
python evaluate_actor_critic.py

# Without rendering (faster)
python evaluate_actor_critic.py --no-render

# Custom number of episodes
python evaluate_actor_critic.py --episodes 50

# Specific model checkpoint
python evaluate_actor_critic.py --model saved_models/actor_critic_model_episode_1000.pth
```

### Evaluate REINFORCE

```bash
python evaluate_reinforce.py --episodes 10
```

### Evaluate REINFORCE with Baseline

```bash
python evaluate_reinforce_baseline.py --episodes 10
```

## Comparing All Agents

To train and compare all three agents:

```bash
python compare_all_agents.py
```

**What happens:**
- Trains all three agents sequentially
- Generates comprehensive comparison plots
- Prints performance statistics for all agents
- Identifies best performing agent

**Output plots:**
- `plots/all_agents_raw_rewards.png` - Episode rewards for all agents
- `plots/all_agents_rolling_average.png` - Rolling average comparison
- `plots/all_agents_statistics.png` - Statistical comparison

**Expected time:** ~30-60 minutes for all three agents

## Understanding the Output

### Training Metrics

During training, you'll see:
```
Episode 50/1200
  Total Reward: 245.00
  Average Reward (last 50): 178.34
  Steps: 245
```

- **Total Reward**: Reward for current episode
- **Average Reward**: Mean over last N episodes
- **Steps**: Number of steps taken

### Evaluation Results

After evaluation:
```
Evaluation Complete!
Number of Episodes: 10
Average Reward: 487.50 ± 12.34
Max Reward: 500.00
Min Reward: 465.00
Average Episode Length: 487.50 ± 12.34
```

### Success Criteria

CartPole-v1 is "solved" when:
- Average reward ≥ 475 over 100 consecutive episodes
- All our agents should achieve this

## Customizing Configuration

Edit `config.py` to change:

```python
# Training duration
NUM_EPISODES = 1200  # Increase for more training

# Network size
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 128

# Learning rates
ACTOR_LEARNING_RATE = 0.0003
CRITIC_LEARNING_RATE = 0.001

# Discount factor
GAMMA = 0.99
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Change in `config.py`:
```python
DEVICE = "cpu"
```

### Issue: Training not converging
**Solutions:**
- Increase `NUM_EPISODES`
- Adjust learning rates
- Check if rewards are improving over time

### Issue: Import errors
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## File Structure After Training

```
Policy Based/
├── saved_models/
│   ├── actor_critic_model_final.pth
│   ├── actor_critic_model_episode_100.pth
│   ├── actor_critic_model_episode_200.pth
│   └── ... (checkpoints every 100 episodes)
├── plots/
│   ├── actor_critic_reward_plot.png
│   ├── actor_critic_rolling_average_reward_plot.png
│   ├── actor_critic_training_summary.png
│   └── ... (similar for other agents)
└── ... (source files)
```

## Next Steps

1. **Experiment with hyperparameters** in `config.py`
2. **Try different environments** by changing `ENV_NAME`
3. **Compare results** with value-based methods in the sibling directory
4. **Analyze the plots** to understand learning dynamics

## Common Workflows

### Quick Test Run
```bash
# Modify config.py: NUM_EPISODES = 100
python train_actor_critic.py
python evaluate_actor_critic.py --no-render --episodes 5
```

### Full Comparison
```bash
python compare_all_agents.py
# Wait for completion (~30-60 minutes)
# Check plots/ directory for comparison graphs
```

### Resume Training
```bash
# Not directly supported - would need to modify training scripts
# Or start fresh with more episodes
```

## Performance Expectations

| Algorithm | Convergence Speed | Final Performance | Variance |
|-----------|------------------|-------------------|----------|
| Actor-Critic | Fast (800-1000 eps) | 475-500 | Low |
| REINFORCE | Slow (1000-1200 eps) | 475-500 | High |
| REINFORCE w/ Baseline | Medium (900-1100 eps) | 475-500 | Medium |

## Getting Help

- Check training plots for learning curves
- Review episode rewards in terminal output
- Ensure all dependencies are properly installed
- Verify GPU/CPU settings if performance issues

## Tips for Best Results

1. **Be patient** - Policy gradient methods can take time to converge
2. **Monitor rolling average** - It's more informative than raw rewards
3. **Save good checkpoints** - Models are saved every 100 episodes
4. **Compare algorithms** - Use `compare_all_agents.py` for comprehensive analysis
5. **Adjust learning rates** - If not learning, try different rates

Happy training! 🚀

# Quick Start Guide - Value-Based RL Implementations

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `torch` - Deep learning framework
- `gymnasium` - Reinforcement learning environments
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `tqdm` - Progress bars

### Step 2: Train the Agent

**Option A: Train DQN (Standard)**
```bash
python train_dqn.py
```

**Option B: Train Double DQN**
```bash
python train_double_dqn.py
```

**Option C: Train SARSA**
```bash
python train_sarsa.py
```

**Option D: Compare All Agents**
```bash
python compare_all_agents.py
```

Training will:
- Run for 1200 episodes (configurable in `config.py`)
- Save models every 100 episodes to `saved_models/`
- Generate training plots in `plots/`
- Print progress every 50 episodes

### Step 3: Evaluate the Trained Agent

**Test DQN:**
```bash
python evaluate_dqn.py
```

**Test Double DQN:**
```bash
python evaluate_double_dqn.py
```

**Test SARSA:**
```bash
python evaluate_sarsa.py
```

**Custom evaluation:**
```bash
python evaluate_dqn.py --model saved_models/dqn_model_episode_1000.pth --episodes 20
```

---

## ⚙️ Customization

### Modify Hyperparameters

Edit `config.py`:

```python
# Example: Train for more episodes with slower exploration decay
NUM_EPISODES = 2000
EPSILON_DECAY = 0.997

# Example: Use larger network
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 128

# Example: Increase batch size
BATCH_SIZE = 128
```

### Change Environment

```python
# In config.py
ENV_NAME = "LunarLander-v2"  # Try different environment
INPUT_DIM = 8  # Update based on new environment
OUTPUT_DIM = 4
```

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `train_dqn.py` | Train standard DQN agent |
| `train_double_dqn.py` | Train Double DQN agent |
| `train_sarsa.py` | Train SARSA agent |
| `evaluate_dqn.py` | Evaluate DQN agent |
| `evaluate_double_dqn.py` | Evaluate Double DQN agent |
| `evaluate_sarsa.py` | Evaluate SARSA agent |
| `compare_all_agents.py` | Compare all three agents |
| `config.py` | All hyperparameters |
| `agents/dqn_agent.py` | DQN implementation |
| `agents/double_dqn_agent.py` | Double DQN implementation |
| `agents/sarsa_agent.py` | SARSA implementation |

---

## 🎯 Expected Results

After training (~1000-1200 episodes):
- Average reward: **450-500** (max is 500 for CartPole-v1)
- Episode length: **500 steps** (environment maximum)
- Success rate: **>95%**

The agent should be able to balance the pole consistently!

---

## 🐛 Troubleshooting

**Issue: Import errors**
```bash
# Solution: Ensure you're in the correct directory
cd "c:\Users\1511l\Desktop\Assignments\Sem V\FAI Project\Multiagent-Systems-Project\Value Based"
```

**Issue: CUDA not available**
```python
# Solution: Edit config.py
DEVICE = "cpu"  # Change from "cuda" to "cpu"
```

**Issue: Slow training**
```python
# Solution: Reduce number of episodes in config.py
NUM_EPISODES = 500  # Instead of 1200
```

**Issue: Agent not learning**
- Check epsilon decay (should start at 1.0 and decay slowly)
- Verify replay buffer is filling (needs at least 64 samples)
- Try adjusting learning rate (increase to 0.001 or decrease to 0.00001)

---

## 📚 Learn More

- **DQN vs Double DQN**: See `README_DQN.md` for detailed comparison
- **Implementation Details**: See `DQN_IMPLEMENTATION_SUMMARY.md`
- **Algorithm Theory**: Check references in `README_DQN.md`

---

## 🎓 Educational Tips

1. **Start by reading** the code in `agents/dqn_agent.py` to understand off-policy learning
2. **Compare with SARSA** in `agents/sarsa_agent.py` to see on-policy learning
3. **Experiment with hyperparameters** in `config.py`
4. **Compare all agents** using `compare_all_agents.py`
5. **Visualize learning** by examining the generated plots
6. **Test on different environments** by modifying `config.py`

---

## 📚 Algorithm Differences

### DQN (Off-Policy)
- Uses experience replay
- Target network for stability
- Learns from past experiences
- Can overestimate Q-values

### Double DQN (Off-Policy)
- Uses experience replay
- Separate action selection and evaluation
- Reduces overestimation bias
- More stable than DQN

### SARSA (On-Policy)
- Learns from current policy
- No experience replay
- More conservative
- Actual action sequence matters

---

## ✅ Success Checklist

- [ ] Dependencies installed
- [ ] Training completed successfully
- [ ] Model saved to `saved_models/`
- [ ] Plots generated in `plots/`
- [ ] Evaluation shows good performance (>450 avg reward)
- [ ] Agent balances pole consistently

Happy Learning! 🎉

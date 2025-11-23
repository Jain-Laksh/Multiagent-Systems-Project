# Comparative Analysis of Reinforcement Learning Agents

<p align="center">
    <img src="cart-pole.gif" alt="CartPole Agent">
</p>

## The Team

| Name | Roll Number | Contribution |
|:-----|:------------|:-------------|
| **Laksh Jain** | 23110185 | Value-Based Agents (DQN, DDQN, SARSA) |
| **Surriya Gokul** | 23110324 | Policy-Based Agents (REINFORCE, Actor-Critic) |
| **Devansh Lodha** | 23110091 | Analysis & Report |

---

## Project Overview

This project presents a comprehensive comparative analysis of **Value-Based** and **Policy-Based** Reinforcement Learning algorithms applied to the classic control problem, **CartPole-v1**.

The objective is to balance a pole attached to a moving cart by applying forces to the left or right. The agent receives a reward of **+1** for every time step the pole remains upright. The episode ends if the pole angle exceeds $\pm 12^\circ$, the cart position exceeds $\pm 2.4$, or the episode length reaches **500**.

We implemented and analyzed six distinct algorithms:
1.  **DQN** (Deep Q-Network)
2.  **Double DQN** (DDQN)
3.  **SARSA** (On-Policy TD Control)
4.  **REINFORCE** (Monte Carlo Policy Gradient)
5.  **REINFORCE with Baseline**
6.  **Actor-Critic**

---

## Theoretical Foundations

The problem is modeled as a **Markov Decision Process (MDP)** defined by the tuple $(S, A, P, R, \gamma)$.

### 1. Value-Based Methods
These methods aim to learn the **Action-Value Function** $Q^\pi(s,a)$, which represents the expected discounted return:

$$ Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t=s, A_t=a \right] $$

**The Bellman Optimality Equation:**
Our DQN agents approximate the optimal $Q^*(s,a)$ by minimizing the Temporal Difference (TD) error:

$$ \text{Loss} = \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 $$

*   **DQN:** Uses a Target Network ($\theta^-$) and Experience Replay to stabilize training.
*   **Double DQN:** Decouples selection and evaluation to reduce overestimation bias:
$$ Y_t^{DDQN} = r + \gamma Q(s', \arg\max_{a} Q(s', a; \theta); \theta^-) $$
*   **SARSA:** An **on-policy** algorithm that updates based on the action actually taken ($\epsilon$-greedy), making it "safer" but potentially slower:
$$ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)] $$

### 2. Policy-Based Methods
These methods directly parameterize the policy $\pi_\theta(a|s)$ and maximize the objective function $J(\theta) = \mathbb{E}[G_0]$.

**The Policy Gradient Theorem:**
We optimize parameters via gradient ascent using the Log-Derivative trick:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right] $$

*   **REINFORCE:** Uses the full Monte Carlo return $G_t$. High variance.
*   **Baseline:** Subtracts a state-value baseline $b(s)$ to reduce variance without introducing bias:
$$ \nabla_\theta J(\theta) \approx \sum \nabla_\theta \log \pi_\theta(a|s) (G_t - b(s)) $$
*   **Actor-Critic:** Replaces the Monte Carlo return with a bootstrapped estimate from a Critic network $V_w(s)$, effectively approximating the Advantage function:
$$ \delta_t = r + \gamma V_w(s') - V_w(s) $$

---

## Environment & Architecture

### State Space (Input Dim: 4)
| Index | Observation | Min | Max |
|:-----:|:------------|:---:|:---:|
| 0 | Cart Position | -4.8 | 4.8 |
| 1 | Cart Velocity | $-\infty$ | $\infty$ |
| 2 | Pole Angle | -0.418 rad | 0.418 rad |
| 3 | Pole Angular Velocity | $-\infty$ | $\infty$ |

### Neural Networks
*   **Value Networks:** MLP (4 $\to$ 64 $\to$ 64 $\to$ 2), ReLU activations.
*   **Policy Networks:** MLP (4 $\to$ 128 $\to$ 128 $\to$ 2), Softmax output.
*   *Note: Policy methods required higher network capacity to avoid catastrophic forgetting.*

---

## Usage

The project is modularized into `Value Based` and `Policy Based` directories.

### To Train an Agent
Navigate to the specific directory and run the training script:

```bash
# Example: Train DQN
cd "Value Based"
python train_dqn.py

# Example: Train Actor-Critic
cd "../Policy Based"
python train_actor_critic.py
```

### To Compare All Agents
Run the comparison script in the respective folder to generate comparative plots:

```bash
python compare_all_agents.py
```

### To Evaluate a Trained Model
```bash
python evaluate_dqn.py --model saved_models/dqn_model_final.pth --no-render
```

---

## Results Summary

*   **Stability:** **Actor-Critic** demonstrated the smoothest convergence, solving the environment efficiently.
*   **Sample Efficiency:** **Double DQN** converged faster than standard DQN by mitigating maximization bias.
*   **Variance:** **REINFORCE** showed high variance, which was successfully mitigated by adding a Baseline.

---

## References

1.  Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2.  Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
3.  Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning.
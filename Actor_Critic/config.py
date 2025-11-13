ENV_CONFIG = {
    'n': 6,
    'start': (0, 0),
    'goal': (5, 5),
    'traps': [(2, 2), (3, 1)],
    'slip_prob': 0.0,
    'max_steps': 100
}

AC_CONFIG = {
    'lr': 3e-3,
    'gamma': 0.99,
    'hidden': 128,
    'device': 'cpu'
}

REINFORCE_CONFIG = {
    'lr': 1e-3,
    'gamma': 0.99,
    'hidden': 128,
    'device': 'cpu'
}

REINFORCE_BASELINE_CONFIG = {
    'lr_policy': 1e-3,
    'lr_value': 1e-3,
    'gamma': 0.99,
    'hidden': 128,
    'device': 'cpu'
}

TRAINING_CONFIG = {
    'n_episodes': 800,
    'seed': 42,
    'log_interval': 160,
    'smooth_window': 10,
    'eval_window': 100
}

VIS_CONFIG = {
    'figsize': (10, 5),
    'grid': True,
    'title': 'Actor-Critic vs REINFORCE on GridWorld'
}

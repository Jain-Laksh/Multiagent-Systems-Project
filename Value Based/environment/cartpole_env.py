import gymnasium as gym


class CartPoleEnv:
    def __init__(self, env_name="CartPole-v1", render_mode="rgb_array"):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = gym.make(env_name, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.input_dim = self.observation_space.shape[0]
        self.output_dim = self.action_space.n
    
    def reset(self, seed=None):
        if seed is not None:
            state, info = self.env.reset(seed=seed)
        else:
            state, info = self.env.reset()
        return state, info
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    def sample_action(self):
        return self.action_space.sample()


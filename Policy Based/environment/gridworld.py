import random


class GridWorld:
    
    def __init__(self, n=5, start=(0, 0), goal=None, traps=None, slip_prob=0.0, max_steps=100):
        self.n = n
        self.start = start
        self.goal = goal if goal is not None else (n-1, n-1)
        self.traps = set(traps) if traps else set()
        self.slip = slip_prob
        self.max_steps = max_steps
        self.reset()

    @property
    def n_states(self):
        return self.n * self.n

    @property
    def n_actions(self):
        return 4

    def reset(self):
        self.pos = tuple(self.start)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return self.pos[0] * self.n + self.pos[1]

    def _inside(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def step(self, action):
        if random.random() < self.slip:
            action = random.randrange(4)

        r, c = self.pos
        
        if action == 0:
            r -= 1
        elif action == 2:
            r += 1
        elif action == 1:
            c += 1
        elif action == 3:
            c -= 1

        if not self._inside(r, c):
            r, c = self.pos

        self.pos = (r, c)
        self.steps += 1

        reward = -0.01
        done = False
        
        if self.pos == self.goal:
            reward = 1.0
            done = True
        elif self.pos in self.traps:
            reward = -1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}

    def render(self):
        grid = [['.' for _ in range(self.n)] for _ in range(self.n)]
        
        r, c = self.pos
        sr, sc = self.start
        gr, gc = self.goal
        
        grid[sr][sc] = 'S'
        grid[gr][gc] = 'G'
        
        for (tr, tc) in self.traps:
            grid[tr][tc] = 'T'
            
        grid[r][c] = 'A'
        
        for row in grid:
            print(' '.join(row))
        print()

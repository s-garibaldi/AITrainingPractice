import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ExpandedGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size=7, n_obstacles=5):
        super().__init__()
        self.size = size
        self.n_obstacles = n_obstacles
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.obstacles = set()
        np.random.seed(seed)
        while len(self.obstacles) < self.n_obstacles:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos != tuple(self.agent_pos) and pos != tuple(self.goal_pos):
                self.obstacles.add(pos)
        return self._get_obs(), {}

    def step(self, action):
        next_pos = list(self.agent_pos)
        if action == 0 and self.agent_pos[0] > 0:
            next_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:
            next_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            next_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:
            next_pos[1] += 1
        if tuple(next_pos) not in self.obstacles:
            self.agent_pos = next_pos
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01
        if tuple(self.agent_pos) in self.obstacles:
            reward = -1.0
            done = True
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def render(self, mode="human"):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 'X'
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        print("\n".join([" ".join(row) for row in grid])) 
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        return self._get_obs(), {}

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:  # right
            self.agent_pos[1] += 1
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def render(self, mode="human"):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        print("\n".join([" ".join(row) for row in grid])) 
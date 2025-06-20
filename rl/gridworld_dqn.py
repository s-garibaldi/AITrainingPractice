import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from gridworld import GridWorldEnv
import gymnasium as gym

class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

def train_gridworld(episodes=500):
    env = GridWorldEnv(size=5)
    obs_size = 1  # Discrete state, so we use a single integer as input
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Action space must be of type Discrete")
    n_actions = env.action_space.n
    policy_net = DQN(obs_size, n_actions)
    target_net = DQN(obs_size, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = deque(maxlen=5000)
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    update_target = 20

    def select_action(state):
        nonlocal epsilon
        if random.random() < epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            state_tensor = torch.FloatTensor([[state]])
            return policy_net(state_tensor).argmax().item()

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor([[s] for s in states])
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor([[s] for s in next_states])
                dones = torch.BoolTensor(dones)
                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q = target_net(next_states).max(1)[0]
                expected_q = rewards + gamma * next_q * (~dones)
                loss = nn.MSELoss()(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if ep % update_target == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    train_gridworld(episodes=100) 
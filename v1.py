import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Environment parameters
num_robots = 3
state_size = 8  # (x, y, vx, vy, obstacle_dist1, obstacle_dist2, obstacle_dist3, obstacle_dist4)
num_actions = 4  # Up, Down, Left, Right
grid_size = 10

# Define a simple environment
class GridWorld:
    def __init__(self, num_robots, grid_size):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.state_size = 8
        self.agents = [self.reset_agent() for _ in range(num_robots)]
        self.targets = [(random.randint(0, grid_size-1), random.randint(0, grid_size-1)) for _ in range(num_robots)]
    
    def reset_agent(self):
        # Reset agent to a random position with random velocity and distances
        x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
        vx, vy = random.uniform(-1, 1), random.uniform(-1, 1)
        obstacles = np.random.rand(4)  # Distance to obstacles
        return [x, y, vx, vy] + list(obstacles)
    
    def step(self, actions):
        rewards = []
        next_states = []
        for i in range(self.num_robots):
            action = actions[i]
            state = self.agents[i]
            reward = 0
            
            if action == 0:  
                state[1] = max(0, state[1] - 1)
            elif action == 1:  
                state[1] = min(self.grid_size - 1, state[1] + 1)
            elif action == 2:  
                state[0] = max(0, state[0] - 1)
            elif action == 3: 
                state[0] = min(self.grid_size - 1, state[0] + 1)

            if (state[0], state[1]) == self.targets[i]:
                reward = 10
                self.targets[i] = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))  # New target
            else:
                reward = -0.1 
            
            next_states.append(state)
            rewards.append(reward)
        
        self.agents = next_states
        return next_states, rewards

# Deep Q-Network (DQN) model
class DQNAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN algorithm with experience replay
class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNAgent(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def store(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state).detach().numpy()[0]
        return np.argmax(q_values)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                target += self.gamma * np.amax(self.model(next_state_tensor).detach().numpy()[0])
            
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            target_f = self.model(state_tensor).detach().numpy()
            target_f[0][action] = target
            
            output = self.model(state_tensor)
            target_tensor = torch.tensor(target_f, dtype=torch.float32)
            loss = self.criterion(output, target_tensor)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = GridWorld(num_robots, grid_size)
agent = DQN(state_size, num_actions)

num_episodes = 1000
batch_size = 32

for episode in range(num_episodes):
    state = env.agents
    total_reward = 0
    
    for step in range(200):
        actions = [agent.act(s) for s in state]
        next_state, rewards = env.step(actions)
        
        for i in range(num_robots):
            agent.store(state[i], actions[i], rewards[i], next_state[i])
        
        state = next_state
        total_reward += sum(rewards)
        
        agent.replay(batch_size)
    
    print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')
    
    if total_reward >= num_robots:  
        print(f'Successful episode at: {episode + 1}')
        break

print("Training completed.")

# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque

# # Define the GridWorld environment
# class GridWorld:
#     def __init__(self, size, num_agents, num_targets, obstacles):
#         self.size = size
#         self.num_agents = num_agents
#         self.num_targets = num_targets
#         self.obstacles = obstacles
#         self.agents = []
#         self.targets = []
#         self.reset()

#     def reset(self):
#         self.agents = [self.random_empty_position() for _ in range(self.num_agents)]
#         self.targets = [self.random_empty_position() for _ in range(self.num_targets)]
#         self.state = self.get_state()
#         return self.state

#     def random_empty_position(self):
#         while True:
#             pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
#             if pos not in self.obstacles and pos not in self.agents and pos not in self.targets:
#                 return pos

#     def get_state(self):
#         state = np.zeros((self.size, self.size))
#         for pos in self.obstacles:
#             state[pos] = -1
#         for pos in self.targets:
#             state[pos] = 1
#         for pos in self.agents:
#             state[pos] = 0.5
#         return state

#     def step(self, actions):
#         rewards = [0] * self.num_agents
#         for i, action in enumerate(actions):
#             self.agents[i] = self.move(self.agents[i], action)
#             if self.agents[i] in self.targets:
#                 rewards[i] = 1
#                 self.targets.remove(self.agents[i])
#                 self.targets.append(self.random_empty_position())
#         self.state = self.get_state()
#         return self.state, rewards

#     def move(self, position, action):
#         if action == 0:  # Up
#             new_position = (position[0] - 1, position[1])
#         elif action == 1:  # Down
#             new_position = (position[0] + 1, position[1])
#         elif action == 2:  # Left
#             new_position = (position[0], position[1] - 1)
#         elif action == 3:  # Right
#             new_position = (position[0], position[1] + 1)
#         else:
#             new_position = position

#         if new_position[0] < 0 or new_position[0] >= self.size or new_position[1] < 0 or new_position[1] >= self.size:
#             return position
#         if new_position in self.obstacles:
#             return position
#         return new_position

# # Define the DQN model
# class DQN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # DQN Agent class
# class DQNAgent:
#     def __init__(self, num_agents, grid_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=0.1, buffer_size=10000, batch_size=64, target_update_freq=10):
#         self.num_agents = num_agents
#         self.grid_size = grid_size
#         self.state_size = grid_size * grid_size  # Flattened grid as input size
#         self.action_size = action_size
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.epsilon = epsilon
#         self.batch_size = batch_size
#         self.target_update_freq = target_update_freq
#         self.buffer_size = buffer_size

#         # Replay buffer
#         self.replay_buffer = deque(maxlen=buffer_size)

#         # Q-network and target network
#         self.q_network = DQN(self.state_size, self.action_size)
#         self.target_network = DQN(self.state_size, self.action_size)
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

#         self.update_target_network()

#         self.steps = 0

#     def update_target_network(self):
#         self.target_network.load_state_dict(self.q_network.state_dict())

#     def choose_action(self, state):
#         state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
#         if random.random() < self.epsilon:
#             return random.randint(0, self.action_size - 1)
#         else:
#             with torch.no_grad():
#                 q_values = self.q_network(state_tensor)
#             return q_values.argmax().item()

#     def store_transition(self, state, action, reward, next_state):
#         self.replay_buffer.append((state, action, reward, next_state))

#     def sample_from_buffer(self):
#         return random.sample(self.replay_buffer, self.batch_size)

#     def learn(self):
#         if len(self.replay_buffer) < self.batch_size:
#             return

#         batch = self.sample_from_buffer()
#         states, actions, rewards, next_states = zip(*batch)

#         states = torch.FloatTensor([s.flatten() for s in states])
#         actions = torch.LongTensor(actions).unsqueeze(1)
#         rewards = torch.FloatTensor(rewards).unsqueeze(1)
#         next_states = torch.FloatTensor([ns.flatten() for ns in next_states])

#         q_values = self.q_network(states).gather(1, actions)

#         next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

#         target_q_values = rewards + self.discount_factor * next_q_values

#         loss = nn.MSELoss()(q_values, target_q_values)
#         print(loss)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         if self.steps % self.target_update_freq == 0:
#             self.update_target_network()

#         self.steps += 1

# size = 10
# num_agents = 3
# num_targets = 3
# obstacles = [(3, 3), (3, 4), (4, 3), (4, 4)]
# env = GridWorld(size, num_agents, num_targets, obstacles)

# num_actions = 4
# agent = DQNAgent(num_agents, size, num_actions)

# num_episodes = 1000
# for episode in range(num_episodes):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     while not done:
#         actions = [agent.choose_action(state) for _ in range(num_agents)]
#         next_state, rewards = env.step(actions)
        
#         for i in range(num_agents):
#             agent.store_transition(state, actions[i], rewards[i], next_state)
#             agent.learn()
        
#         state = next_state
#         total_reward += sum(rewards)
#         if total_reward >= env.num_targets:
#             done = True

#     print(f'Episode {episode}, Total Reward: {total_reward}')

# print("Training completed.")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

num_robots = 3
state_size = 8      # (x, y, vx, vy, obstacle_dist1, obstacle_dist2, obstacle_dist3, obstacle_dist4)
num_actions = 4     # Up, Down, Left, Right
num_samples = 1000  # Number of data samples to generate

def generate_dummy_data(num_samples, num_robots, state_size):
    states = []
    actions = []
    for _ in range(num_samples):
        for _ in range(num_robots):
            state = np.random.rand(state_size)
            action = np.random.randint(0, num_actions)  # Random action (0 to 3)
            states.append(state)
            actions.append(action)
    return np.array(states), np.array(actions)

states, actions = generate_dummy_data(num_samples, num_robots, state_size)

class DQNAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = state_size  
output_size = num_actions  
model = DQNAgent(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

states_tensor = torch.tensor(states, dtype=torch.float32)
actions_tensor = torch.tensor(actions, dtype=torch.long)

num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(states_tensor), batch_size):
        batch_states = states_tensor[i:i+batch_size]
        batch_actions = actions_tensor[i:i+batch_size]
        
        predictions = model(batch_states)
        
        loss = criterion(predictions, batch_actions)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(states_tensor):.4f}')

print("Training completed.")

test_state = np.random.rand(state_size)
test_state_tensor = torch.tensor(test_state, dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_action = model(test_state_tensor.unsqueeze(0)).argmax().item()

print(f'Predicted Action: {predicted_action}')

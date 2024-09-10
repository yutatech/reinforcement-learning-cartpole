import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import random
import copy

import torch
from torch import nn

GAMMA = 0.97
LEARNING_RATE = 0.001
MEMORY_SIZE = 200
BATCH_SIZE = 50

# 環境の生成 
trigger = lambda t: t % 100 == 0 
env = RecordVideo(gym.make('CartPole-v1', render_mode="rgb_array"), './video/', episode_trigger=trigger)
# env = gym.make('CartPole-v1', render_mode="rgb_array")

main_Q_net = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 100), nn.ReLU(),
    nn.Linear(100, 100), nn.ReLU(),
    nn.Linear(100, 100), nn.ReLU(),
    nn.Linear(100, env.action_space.n)
)

loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(main_Q_net.parameters(), lr=LEARNING_RATE)
# optimizer = optim.RMSprop(Q.parameters(), lr=0.00015, alpha=0.95, eps=0.01)

target_Q_net = copy.deepcopy(main_Q_net)

class EpsilonGreedy:
  def __init__(self, random_ratio = 0.1):
    self.random_ratio = random_ratio
  
  def NextAction(self, epsilon = None):
    if epsilon is None:
      epsilon = self.random_ratio
    if random.random() <= epsilon:
      return 'random'
    else:
      return 'optimum'
    
epsilon_greedy = EpsilonGreedy(1)
epsilon = 1
step_counter = 0

memory = []

for episode in range(801):
  prev_observation = env.reset()[0]
  for t in range(200):
    # 現在の状況を表示させる
    env.render()
    
    # 次の行動を決定する
    if epsilon_greedy.NextAction(epsilon) == 'random':
      prev_action = env.action_space.sample()
    else:
      main_Q_net.eval()
      prev_action = int(main_Q_net.forward(torch.FloatTensor(prev_observation)).argmax())
    # print('actions', main_Q_net.forward(torch.FloatTensor(prev_observation)))
    
    observation, reward, terminated, truncated, info = env.step(prev_action)
    if terminated:
      reward = -1
      
    memory.append((prev_observation, prev_action, observation, reward, terminated))

    if len(memory) > MEMORY_SIZE:
      memory.pop(0)
      
    if len(memory) == MEMORY_SIZE and step_counter % 10 == 0:
      for i in range(4):
        batch = random.sample(memory, BATCH_SIZE)
        
        prev_observation_batch = torch.FloatTensor([x[0] for x in batch])
        prev_action_batch = [x[1] for x in batch]
        observation_batch = torch.FloatTensor([x[2] for x in batch])
        reward_batch = [x[3] for x in batch]
        terminated_batch = [x[4] for x in batch]
        
        prev_action_q_batch = main_Q_net(prev_observation_batch)
        
        target_Q_net.eval()
        target_q_batch = copy.deepcopy(prev_action_q_batch.data.numpy())
        
        for i in range(BATCH_SIZE):
          if terminated_batch[i]:
            target_q_batch[i][int(prev_action_batch[i])] = reward_batch[i]
          else:
            target_q_batch[i][int(prev_action_batch[i])] = reward_batch[i] + GAMMA * target_Q_net.forward(observation_batch[i]).max()
      
        main_Q_net.train()
        optimiser.zero_grad()
        loss = loss_fn(prev_action_q_batch, torch.FloatTensor(target_q_batch))
        loss.backward()
        optimiser.step()
        print(loss.item())
      
    prev_observation = copy.deepcopy(observation)
    
    step_counter += 1
    if step_counter % 200 == 0:
      epsilon -= 0.01
      if epsilon < 0.1:
        epsilon = 0.1
    if step_counter % 20 == 0:
      target_Q_net = copy.deepcopy(main_Q_net)
    
    if terminated:
      break
    
    

# 環境を閉じる
env.close()
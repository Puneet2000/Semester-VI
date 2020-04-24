import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from collections import namedtuple
import random
from PIL import Image
import matplotlib.pyplot as plt

def plot(steps,mean_rewards,best_mean_rewards):
    plt.figure(figsize=(14,7)) 

    plt.plot(steps,mean_rewards,label='Mean Reward')
    plt.plot(steps,best_mean_rewards,label='Best Mean Reward')
    plt.title('Training...')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()

    plt.savefig('dqn_pong.png')

class DQN(nn.Module):
    def __init__(self, in_channels=3, n_actions=3):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    # eps_threshold = 0.1
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    for _ in range(1):
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
        rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(device)
        

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

transform1  =  T.Compose([T.ToPILImage(), T.CenterCrop(160), T.Grayscale()])
transform2 = T.Compose([T.Resize((84,84)),T.ToTensor()])
def get_state(obs, prev_state=None):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    # print(state.shape)
    state = transform1(state)
    state =  T.functional.adjust_contrast(state,10)
    state = transform2(state)
    if prev_state is None:
        state = torch.cat([state,state,state],0)
    else:
        state  = torch.cat([prev_state.squeeze(0)[-2:],state],0)
    return state.unsqueeze(0)

action_map = {0:0,1:2,2:3}
def train(env, n_episodes, render=False):
    steps = []
    rewards = []
    mean_rewards = []
    
    best_mean_rewards = []
    best_mean_reward = -21.0
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        torchvision.utils.save_image(state, 'a.png')
        total_reward = 0.0
        for t in count():
            action = select_action(state)
            # action = action.item()
            # print(action)
            if render:
                env.render()

            obs, reward, done, info = env.step(action_map[action.item()])

            total_reward += reward

            if not done:
                next_state = get_state(obs, state)
                torchvision.utils.save_image(next_state, 'b.png')
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()
            
            if (steps_done+1) % TARGET_UPDATE == 0:
              target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        rewards.append(total_reward)
        if episode >= 10:
            steps.append(steps_done)
            mean_reward = np.mean(rewards[-10:])
            mean_rewards.append(mean_reward)
            best_mean_reward = max(mean_reward,best_mean_reward)
            best_mean_rewards.append(best_mean_reward)
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
        d = {'steps':steps, 'mean_rewards': mean_rewards, 'best_mean_rewards':best_mean_rewards}
        np.save('/content/drive/My Drive/dqn_results.npy',d)
        # plot(steps,mean_rewards,best_mean_rewards)
        torch.save(policy_net, "/content/drive/My Drive/dqn_pong.pt")
        torch.save(target_net, "/content/drive/My Drive/dqn_pong_t.pt")

    env.close()
    return


if __name__ == '__main__':
    # set device
    device = torch.device("cuda")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.9
    EPS_START = 1.0
    EPS_END = 0.02
    EPS_DECAY = 25000
    TARGET_UPDATE = 1000
    INITIAL_MEMORY = 1000
    MEMORY_SIZE = 20* INITIAL_MEMORY
    NUM_EPISODES = 2000

    # create networks
    policy_net = DQN(n_actions=3).to(device)
    target_net = DQN(n_actions=3).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # policy_net =  torch.load("/content/drive/My Drive/dqn_pong.pt")
    # target_net =  
    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(),lr=1e-4)

    steps_done = 0

    # create environment
    env = gym.make('Pong-v0')
    # env = make_env(env)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    train(env, NUM_EPISODES)
    
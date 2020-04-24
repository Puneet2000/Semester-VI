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
import torchvision.transforms as T

from collections import namedtuple
import random
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def plot(steps,mean_rewards,best_mean_rewards,policy):
    fig = plt.figure(figsize=(10,5)) 
    plt1 = fig.add_subplot(121)
    plt2 = fig.add_subplot(122) 

    plt1.plot(steps,mean_rewards,label='Mean Reward')
    plt1.plot(steps,best_mean_rewards,label='Best Mean Reward')
    plt1.set_title('Training...')
    plt1.set_xlabel('Steps')
    plt1.set_ylabel('Reward')
    plt1.grid()
    plt1.legend()

    X = np.random.uniform(-1.2, 0.6, 1000)
    Y = np.random.uniform(-0.07, 0.07, 1000)
    Z = []
    for i in range(len(X)):
        _, temp = torch.max(policy(torch.from_numpy(np.array([X[i],Y[i]])).type(torch.FloatTensor)), dim =-1)
        z = temp.item()
        Z.append(z)
    Z = pd.Series(Z)
    colors = {0:'blue',1:'lime',2:'red'}
    colors = Z.apply(lambda x:colors[x])
    labels = ['Left','Right','Nothing']

    plt.set_cmap('brg')
    surf = plt2.scatter(X,Y, c=Z)
    plt2.set_xlabel('Position')
    plt2.set_ylabel('Velocity')
    plt2.set_title('Policy')
    recs = []
    for i in range(0,3):
         recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    plt2.legend(recs,labels,loc=4,ncol=3)

    plt.savefig('dqn_mc.png')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 1)

class DQN(nn.Module):
    def __init__(self):

        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 200,bias=False)
        self.fc2 = nn.Linear(200, 3,bias=False)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

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
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    for index in range(100):
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
        rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8)
        
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

# transform  =  T.Compose([T.ToPILImage(), T.Grayscale(),T.Resize((64,64)), T.ToTensor()])
def get_state(obs, prev_state=None):
    state = np.array(obs)
    state = torch.from_numpy(state).type(torch.FloatTensor)
    return state.unsqueeze(0)

def train(env, n_episodes, render=False):
    steps = []
    rewards = []
    mean_rewards = []
    
    best_mean_rewards = []
    best_mean_reward = -200.0

    successes = 0
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        # print(state.shape)
        total_reward = 0.0
        for t in count():
            action = select_action(state)

            if render:
                env.render()
            action = action.item()
            obs, reward, done, info = env.step(action)
            
            total_reward += reward

            reward = obs[0] + 0.5
            if obs[0] >= 0.5:
                reward += 1.0
            

            if not done:
                next_state = get_state(obs, state)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action, next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

            if done:
                if obs[0] >= 0.5:
                    successes +=1
                break
        if (episode+1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        rewards.append(total_reward)
        if episode >= 5:
            steps.append(steps_done)
            mean_reward = np.mean(rewards[-5:])
            mean_rewards.append(mean_reward)
            best_mean_reward = max(mean_reward,best_mean_reward)
            best_mean_rewards.append(best_mean_reward)
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t Success {}'.format(steps_done, episode, t, total_reward, successes))
    env.close()
    d = {'steps':steps, 'mean_rewards': mean_rewards, 'best_mean_rewards':best_mean_rewards}
    np.save('./dqn8_results.npy',d)
    plot(steps,mean_rewards,best_mean_rewards,policy_net)
    return


if __name__ == '__main__':
    # set device
    device = torch.device("cpu")

    # hyperparameters
    BATCH_SIZE = 8
    GAMMA = 0.9
    EPS_START = 1.0
    EPS_END = 0.02
    EPS_DECAY = 1000
    TARGET_UPDATE = 5
    INITIAL_MEMORY = 1000
    MEMORY_SIZE = 1* INITIAL_MEMORY
    NUM_EPISODES = 100

    # create networks
    policy_net = DQN().to(device)
    # policy_net.apply(weights_init)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    steps_done = 0

    # create environment
    env = gym.make('MountainCar-v0')
    # env = make_env(env)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    train(env, NUM_EPISODES)

    torch.save(policy_net, "./dqn_mc.pt")
    # policy_net = torch.load("dqn_pong_model")
    # test(env, 1, policy_net, render=False)
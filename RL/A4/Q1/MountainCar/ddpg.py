import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
from collections import deque
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

def plot(rewards,avg_rewards):
    fig = plt.figure(figsize=(10,5)) 

    plt.plot(rewards,label='Episode Reward')
    plt.plot(avg_rewards,label='Average Reward')
    plt.title('Training...')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()

    plt.savefig('ddpg.png')

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class GaussNoise(object):
    def __init__(self, action_space, mu=0.0, sigma=0.6, decay_period=50000.0):
        self.mu           = mu
        self.sigma        = sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        pass
    
    def get_action(self, action, t=0): 
        noise = np.exp(-t/self.decay_period)*np.random.normal(self.mu,self.sigma,self.action_dim)
        return np.clip(action + noise, self.low, self.high)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    action = actor(state)
    action = action.detach().numpy()[0,0]
    return action
    
def optimize_model():
    states, actions, rewards, next_states, _ = memory.sample(BATCH_SIZE)
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    
    # Critic loss        
    Qvals = critic(states, actions)
    next_actions = target_actor(next_states)
    next_Q = target_critic(next_states, next_actions.detach())
    Qprime = rewards + GAMMA* next_Q
    critic_loss = F.mse_loss(Qvals, Qprime)

    # Actor loss
    policy_loss = -critic(states, actor(states)).mean()
        
    # update networks
    actor_optimizer.zero_grad()
    policy_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward() 
    critic_optimizer.step()

    # update target networks 
    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data *TAU + target_param.data * (1.0 - TAU))
       
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data *TAU + target_param.data * (1.0 - TAU))

# transform  =  T.Compose([T.ToPILImage(), T.Grayscale(),T.Resize((64,64)), T.ToTensor()])
def get_state(obs, prev_state=None):
    state = np.array(obs)
    state = torch.from_numpy(state).type(torch.FloatTensor)
    return state.unsqueeze(0)

def train(env, n_episodes):
    rewards = []
    avg_rewards = []
    t = 0
    for episode in range(n_episodes):
        state = env.reset()
        noise.reset()
        episode_reward = 0
    
        for step in range(1000):
            action = select_action(state)
            action = noise.get_action(action, t)
            t += 1
            new_state, reward, done, _ = env.step(action) 
            memory.push(state, action, reward, new_state, done)
        
            if len(memory) > BATCH_SIZE:
                optimize_model()        
        
            state = new_state
            episode_reward += reward

            if done:
                break
        print("Episode: {}, Reward: {:3f}".format(episode, episode_reward))
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
    plot(rewards,avg_rewards)
    np.save('./res_gauss.npy',{0:rewards,1:avg_rewards})
    return


if __name__ == '__main__':

    # hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    MEMORY_SIZE = 50000
    NUM_EPISODES = 200
    TAU = 0.01
    env = gym.make('MountainCarContinuous-v0')
    # noise = OUNoise(env.action_space)
    noise = GaussNoise(env.action_space)

    # create networks
    actor = Actor(env.observation_space.shape[0], 256, env.action_space.shape[0])
    target_actor = Actor(env.observation_space.shape[0], 256, env.action_space.shape[0])
    target_actor.load_state_dict(actor.state_dict())

    critic = Critic(env.observation_space.shape[0] + env.action_space.shape[0], 256 )
    target_critic = Critic(env.observation_space.shape[0] + env.action_space.shape[0], 256 )
    target_critic.load_state_dict(critic.state_dict())


    # setup optimizer
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    train(env, NUM_EPISODES)
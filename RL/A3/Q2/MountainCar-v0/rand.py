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

def plot(steps,mean_rewards,best_mean_rewards):
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
        z = random.randrange(3)
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

    plt.savefig('random_mc.png')


def select_action(state):
    global steps_done
    steps_done += 1
    return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

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
            next_state =  get_state(obs)
            total_reward += reward

            state = next_state

            if done:
                if obs[0] >= 0.5:
                    successes +=1
                break
        rewards.append(total_reward)
        if episode >= 5:
            steps.append(steps_done)
            mean_reward = np.mean(rewards[-5:])
            mean_rewards.append(mean_reward)
            best_mean_reward = max(mean_reward,best_mean_reward)
            best_mean_rewards.append(best_mean_reward)
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t Success {}'.format(steps_done, episode, t, total_reward, successes))
    env.close()
    plot(steps,mean_rewards,best_mean_rewards)
    return


if __name__ == '__main__':
    # set device
    device = torch.device("cpu")
    NUM_EPISODES = 100

    steps_done = 0

    # create environment
    env = gym.make('MountainCar-v0')
    
    # train model
    train(env, NUM_EPISODES)
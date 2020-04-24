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

def plot(rewards):
    fig = plt.figure(figsize=(10,5)) 

    plt.plot(rewards,label='Mean Reward')
    plt.title('Training...')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()

    plt.savefig('rand.png')


def select_action(state):
    global steps_done
    steps_done += 1
    return np.random.uniform(env.action_space.low[0],env.action_space.high[0])

def get_state(obs, prev_state=None):
    state = np.array(obs)
    state = torch.from_numpy(state).type(torch.FloatTensor)
    return state.unsqueeze(0)

def train(env, n_episodes):
    rewards = []

    successes = 0
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)
            obs, reward, done, info = env.step([action])
            next_state =  get_state(obs)
            total_reward += reward

            state = next_state

            if done:
                if obs[0] >= 0.5:
                    successes +=1
                break
        rewards.append(total_reward)
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t Success {}'.format(steps_done, episode, t, total_reward, successes))
    env.close()
    plot(rewards)
    return


if __name__ == '__main__':
    # set device
    device = torch.device("cpu")
    NUM_EPISODES = 100

    steps_done = 0

    # create environment
    env = gym.make('MountainCarContinuous-v0')
    print(env.action_space.high)
    
    # train model
    train(env, NUM_EPISODES)
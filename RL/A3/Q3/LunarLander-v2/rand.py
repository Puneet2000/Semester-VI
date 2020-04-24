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
from torch.distributions import Categorical

from collections import namedtuple
import random
from PIL import Image
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env",type=str, default = "LunarLander-v2")
parser.add_argument("--iterations",type=int, default=150)
parser.add_argument("--batch",type=int, default=64)


def plot(rewards):
    plt.figure(figsize=(10,5)) 

    plt.plot(rewards,label='Mean Reward')
    plt.title('Training...')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid()

    plt.savefig('pg_random.png')


def select_action(state):
    return random.randrange(0,env.action_space.n)
    
def get_state(obs):
    state = np.array(obs)
    state = torch.from_numpy(state).type(torch.FloatTensor)
    return state.unsqueeze(0)

def train(env):
    rewards = []

    for epoch in range(ITERATIONS):
        episode_rewards_batch = []
        for episode in range(BATCH_SIZE):
            obs = env.reset()
            state = get_state(obs)
            episode_rewards = []
            policy_history = []
            total_reward = 0.0
            for t in count():
                action= select_action(state)
                obs, reward, done, info = env.step(action)
                state  =  get_state(obs)
                episode_rewards.append(reward)

                if done:
                    break
            episode_rewards_batch.append(episode_rewards)
        rewards.append(np.mean([np.sum(er) for er in episode_rewards_batch]))
        print('Iteration: {} \t Total reward: {}'.format(epoch,rewards[-1]))  
        
    env.close()
    plot(rewards)
    return


if __name__ == '__main__':
    # set device
    device = torch.device("cpu")
    args = parser.parse_args()

    ITERATIONS = args.iterations
    BATCH_SIZE = args.batch
    env = gym.make(args.env)
    train(env)
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
    plt.figure(figsize=(10,5)) 

    plt.plot(steps,mean_rewards,label='Mean Reward')
    plt.plot(steps,best_mean_rewards,label='Best Mean Reward')
    plt.title('Training...')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()

    plt.savefig('random.png')

def select_action(state):
    global steps_done
    steps_done += 1
    return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)


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
        # torchvision.utils.save_image(state, 'a.png')
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
    plot(steps,mean_rewards,best_mean_rewards)

    env.close()
    return


if __name__ == '__main__':
    # set device
    device = torch.device("cpu")

    NUM_EPISODES = 50

    steps_done = 0

    # create environment
    env = gym.make('Pong-v0')

    # train model
    train(env, NUM_EPISODES)
    
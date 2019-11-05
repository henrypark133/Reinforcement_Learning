import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float).ravel(), [80*80])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(80*80, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(256, 2)

        # Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform')

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.l2(x)
        return F.softmax(action_scores, dim=1)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    Reward = 0
    policy_loss = []
    value_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        Reward = r + discount_rate * Reward
        rewards.insert(0, Reward)

    # possible_values_0.append(discount_rate * val[possible_next_state])

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / rewards.std()
    value = torch.tensor(rewards[0])
    for (log_prob, reward) in zip(policy.saved_log_probs, rewards):
        advantage = reward - value
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.mse_loss(value, torch.tensor([reward])))

    policy_optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    policy_loss.backward()
    policy_optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# export cuda_visible_devices=0

env = gym.make('Pong-v0')
env.seed(1)
torch.manual_seed(1)

policy = Policy()
policy = policy.to(device)
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
discount_rate = 0.95

def main():
    avg_reward = []
    for i_episode in range(1000):
        state = env.reset()
        epi_reward = 0
        for t in range(1, 1000):
            state = preprocess(state)
            action = select_action(state)
            state, reward, done, _ = env.step(action+2)
            policy.rewards.append(reward)
            epi_reward += reward
            if done:
                break

        finish_episode()
        avg_reward.append(epi_reward)
        if i_episode % 10 == 0:
            print(epi_reward)
            
    print(avg_reward)

    plt.plot(range(len(avg_reward)), avg_reward)
    plt.title("Average Reward for Each Epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.ylim((min(avg_reward)-2,max(avg_reward)+2))
    plt.xlim((0,len(avg_reward)-1))
    plt.savefig('pong.png')

    # state, ep_reward = env.reset(), 0
    # goal_steps = 200
    # for t in range(goal_steps):
    #     env.render()
    #     action = select_action(state)
    #     state, reward, done, _ = env.step(action)
    #     policy.rewards.append(reward)
    #     ep_reward += reward
    #     if done:
    #         break


if __name__ == '__main__':
    main()


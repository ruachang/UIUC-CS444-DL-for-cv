import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN, DQN_LSTM
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        
    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state, device):
        if np.random.rand() <= self.epsilon:
            ### TODO #### 
            # Choose a random action
            a = np.random.choice(self.action_size)
            a = torch.tensor(a).to(device)
        else:
            ### TODO ####
            # Choose the best action
            with torch.no_grad():
                state = torch.from_numpy(state).unsqueeze(dim=0).to(device)
                actions_q = self.policy_net(state)
                best_action = torch.argmax(actions_q)
                a = best_action
                # a = torch.tensor(a).to(device)
        
        return a

    def huber_loss(self, loss):
        batch_loss = torch.where(torch.abs(loss) <= 1, 0.5 * loss * loss, self.epsilon * (torch.abs(loss) - 0.5 * self.epsilon))
        return batch_loss
        # if torch.abs(loss) <= self.epsilon:
        #     return 0.5 * loss * loss 
        # else:
        #     return self.epsilon * (torch.abs(loss) - 0.5 * self.epsilon)
    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame, device):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = (np.float32(history[:, 1:, :, :]) / 255.)
        next_states = torch.from_numpy(next_states).to(device)
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8).to(device)


        # Compute Q(s_t, a), the Q-value of the current state
        ### TODO ####
        Q_current = self.policy_net(states)
        Q_current = Q_current[range(len(actions)), actions]
        # Compute Q function of next state
        ### TODO ####
        with torch.no_grad():
            Q_next = self.policy_net(next_states)
            # Find maximum Q-value of action at next state from policy net
            ### TODO ####
            Q_next_max = torch.max(Q_next, dim=1)[0]
            # print(Q_next_max.shape, mask.shape)
            Q_next_max_mask = Q_next_max * mask
        self.optimizer.zero_grad()
        # Compute the Huber Loss
        ### TODO ####
        loss = rewards + self.discount_factor * Q_next_max_mask - Q_current
        loss = self.huber_loss(loss)
        loss = torch.mean(loss)
        # Optimize the model, .step() both the optimizer and the scheduler!
        ### TODO ####
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

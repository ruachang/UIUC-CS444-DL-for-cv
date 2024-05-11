from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from agent import Agent
from memory import ReplayMemoryLSTM
from model import DQN_LSTM
from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_LSTM():
    def __init__(self, action_size):
        # Generate the memory
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000
        self.memory = ReplayMemoryLSTM()

        # Create the policy net
        self.policy_net = DQN_LSTM(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)           

    # after some time interval update the target net to be same with policy net
    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state, hidden=None):
        ### CODE ###
        # Similar to that for Agent
        # You should pass the state and hidden through the policy net even when you are randomly selecting an action so you can get the hidden state for the next state
        # We recommend the following outline:
        # 1. Pass the state and hidden through the policy net. You should pass train=False to the forward function of the policy net here becasue you are not training the policy net here
        # 2. If you are randomly selecting an action, return the random action and policy net's hidden, otherwise return the policy net's action and hidden
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(dim=0).to(device)
            actions_q, hidden = self.policy_net(state, hidden, train=False)
        if np.random.rand() <= self.epsilon:
            a = np.random.choice(self.action_size)
            a = torch.tensor(a).to(device)
            # print(a.cpu(), a.type())
        else:
            a = torch.argmax(actions_q)
            # print(a.cpu(), a.type())
        return a, hidden
    def huber_loss(self, loss):
        batch_loss = torch.where(torch.abs(loss) <= 1, 0.5 * loss * loss, self.epsilon * (torch.abs(loss) - 0.5 * self.epsilon))
        return batch_loss
    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :lstm_seq_length, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).to(device)
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8).to(device)

        ### All the following code is nearly same as that for Agent

        # Compute Q(s_t, a), the Q-value of the current state
        # You should hidden=None as input to policy_net. It will return lstm_state and hidden. Discard hidden. Use the last lstm_state as the current Q values
        ### CODE ####
        Q_current, _ = self.policy_net(states, hidden=None, train=True)
        Q_current = Q_current[range(len(actions)), actions]
        # Compute Q function of next state
        # Similar to previous, use hidden=None as input to policy_net. And discard the hidden returned by policy_net
        ### CODE ####
        with torch.no_grad():
            Q_next, _ = self.policy_net(next_states, hidden=None, train=True)
            # next_best_action = torch.argmax(Q_next, dim=1)
            # Q_next_best, _ = self.target_net(next_states)
            # Q_next_max = Q_next_best[range(len(next_best_action)), next_best_action]
            
            Q_next_max = torch.max(Q_next, dim=1)[0]
            Q_next_max_mask = Q_next_max * mask
        self.optimizer.zero_grad()
        # Find maximum Q-value of action at next state from policy net
        # Use the last lstm_state as the Q values of next state
        ### CODE ####
        loss = rewards + self.discount_factor * Q_next_max_mask - Q_current
        loss = self.huber_loss(loss)
        loss = torch.mean(loss)
        # Optimize the model, .step() both the optimizer and the scheduler!
        ### TODO ####
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-10, 10)
        self.optimizer.step()
        self.scheduler.step()

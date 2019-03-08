import random
import torch
import numpy as np

from model import DQN_MODEL


BUFFER_SIZE = int(1e5)
LR = 3e-4
TAU = 1e3
GAMMA = 0.99
UPDATE_EVERY = 4


class DQN_AGENT:
    def __init__(self, state_size, action_size, seed):
        """
        Initialize DQN agent to defaults on creation.

        params :
            state_size     - environment state space size.
            action_size    - environment action space size.
            seed           - random number generator seed value.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.time_steps = 0
        self.device = torch.device('cpu') # CPU's are faster for Unity Environments

        self.learning_network = DQN_MODEL()
        self.target_network = DQN_MODEL()
        self.replay_memory = REPLAY_MEMORY()
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.learning_network.parameters(), lr=LR)


    def step(self, state, action, reward, next_state, done):
        """
        DQN Agent takes a step and adds the experience to replay memory.
        If condition to learn is satisfied, then agent learns from sampled
        experiences from the replay memory.

        params :
            state        - current environment state.
            action       - action chosen by agent in current state.
            reward       - reward obtained from environment.
            next_state   - next state of environment, a result of current state-action combination.
            done         - whether episode is finished or not.
        """
        self.replay_memory.add(state, action, reward, next_state, done)
        self.time_steps += 1

        if self.time_steps % UPDATE_EVERY == 0:
            experiences = self.replay_memory.sample()
            self.learn(experiences)

        self.target_network_update(self.learning_network, self.target_network, TAU)


    def act(self, state):
        """
        Take in enviroment state and return action to be taken by agent based
        on current policy followed by agent.

        params :
            state - current enviroment state
        """
        pass


    def learn(self):
        pass


    def target_network_update(self, learning_network, target_network, TAU):
        pass



class REPLAY_MEMORY:
    def __init__(self):
        pass


    def add(self):
        pass


    def sample(self):
        pass
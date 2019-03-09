import random
import torch
import numpy as np
from collections import namedtuple, deque

from model import DQN_MODEL


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
LR = 5e-2
TAU = 1e-1
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
        self.device = torch.device(DEVICE) # CPU's are faster for Unity Environments
        self.seed = random.seed(seed)

        self.learning_network = DQN_MODEL(self.state_size, self.action_size, seed).to(self.device)
        self.target_network = DQN_MODEL(self.state_size, self.action_size, seed).to(self.device)
        self.replay_memory = REPLAY_MEMORY(BUFFER_SIZE, BATCH_SIZE, seed)
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

        if self.time_steps % UPDATE_EVERY == 0 and len(self.replay_memory) > BATCH_SIZE:
            experiences = self.replay_memory.sample()
            self.learn(experiences, GAMMA)


    def act(self, state, eps):
        """
        Take in enviroment state and return action to be taken by agent based
        on current policy followed by agent.

        params :
            state - current enviroment state (numpy.ndarray).
            eps   - epsilon value.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.learning_network.eval()
        with torch.no_grad():
            action_values = self.learning_network(state)
        self.learning_network.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))


    def learn(self, experiences, GAMMA):
        """
        Learn from sampled experiences and update learning_network accordingly.

        params :
            experiences - sampled experiences from replay memory.
            GAMMA       - gamma value (importance given to future rewards).
        """
        states, actions, rewards, next_states, dones = experiences
        targets = self.target_network(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + (GAMMA * targets * (1-dones))
        chosen_actions = self.learning_network(states).gather(1, actions)

        loss = self.loss_function(chosen_actions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_network_update(self.learning_network, self.target_network, TAU)


    def target_network_update(self, learning_network, target_network, TAU):
        """
        Update target network of DQN Agent to provide fixed targets for
        learning.

        params :
            learning_network - weights to use for updating target network.
            target_network   - target network whose weights to update.
            TAU              - time steps for updating target network weights.
        """
        for learning_params, target_params in zip(learning_network.parameters(), target_network.parameters()):
            target_params.data.copy_(TAU*learning_params.data + (1-TAU)*target_params.data)



class REPLAY_MEMORY:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialize replay memory buffer to defaults on creation.

        params :
            buffer_size    - replay memory size.
            batch_size     - batch size of sampled experiences to return.
            seed           - random generator seed value.
        """
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.device = torch.device(DEVICE)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])


    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to replay memory by storing passed values.

        params :
            state        - current environment state.
            action       - action chosen by agent in current state.
            reward       - reward obtained from environment.
            next_state   - next state of environment, a result of current state-action combination.
            done         - whether episode is finished or not.
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)


    def sample(self):
        """
        Return BATCH_SIZE number of sampled experiences from replay memory.
        """
        samples = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """
        Return length of replay memory.
        """
        return len(self.memory)

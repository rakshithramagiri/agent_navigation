import numpy as np
import torch

class DQN_MODEL(torch.nn.Module):
    def __init__(self, state_size, action_size, seed, hidden=512):
        """
        Declare and Initialize Deep Neural Net.

        params :
            state_size     - state space size.
            action_size    - action space size.
            seed           - random number generator seed value.
            hidden         - number of hidden units.
        """
        super(DQN_MODEL, self).__init__()

        self.hidden = hidden
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, int(self.hidden//4)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.hidden//4), self.action_size)
        )

        self.softmax = torch.nn.Softmax(dim=1)



# Banana Picking Agent

Train intelligent agents to navigate Unity's Bananas environment to pick up **yellow** bananas while avoiding **blue** bananas.

## Environment

### Goal
The agent must learn to move and collect as many yellow bananas as possible while avoiding blue bananas.

### Rewards
Reward function for agent is `+1` for each **yellow** banana collected and `-1` for each **blue** banana collected.

### Environment Details
This is an episodic task. The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - move forward.
- `1` - move backward.
- `2` - turn left.
- `3` - turn right.

### Solved
Environment is considered as solved, when our agent gets an average score of `+13` over 100 consecutive episodes.

## Getting Started

1. Clone this repository to your local machine using `git clone`.
```
git clone https://github.com/rakshithramagiri/agent_navigation.git
```

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. After download, extract the **zip** file and place extracted folder in the root of **agent_navigation** repo.

4. Install **unityagents** package in your environment - `pip install unityagents`

5. Ensure following packages are installed in your environment before proceeding with training or running demo.
    - PyTorch - [click here](https://pytorch.org/get-started/locally/)
    - Numpy - `pip install numpy`
    - Matplotlib - `pip install matplotlib`


## Instructions

- To train your own agent, run all cells of notebook`[TRAIN] agent_navigation.ipynb`.

- To watch demo of a trained agent, run cells in `[DEMO] agent_navigation.ipynb` notebook.

> NOTE - Change path of `file_name` variable in both notebooks to reflect your Unity Environment files.
```
env = UnityEnvironment(file_name= < YOUR UNITY ENV PATH HERE >)
```

## Algorithm
- This repo uses **DQN** alogorithm to solve Unity's Banana Collector environment.

- The network architecture used is a simple, fully-connected neural network with 2 hidden layers of sizes, 128 and 256 units. Model definition can be found in `model.py` file.

- Hyperparameter values chosen are as follows :

| Hyperparameter | Value |
| -------------- | ------ |
| Learning Rate | 5e-4 |
| Gamma | 0.99 |
| Buffer Size | 1e5 |
| Batch Size | 32 |

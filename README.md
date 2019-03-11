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

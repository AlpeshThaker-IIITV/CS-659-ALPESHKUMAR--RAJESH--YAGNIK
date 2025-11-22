# epsilon_greedy_nonstat.py
"""
Modified Epsilon-Greedy Agent for Non-Stationary Bandit
-------------------------------------------------------
This script defines an epsilon-greedy agent to interact with a non-stationary
10-armed bandit. It uses a constant step-size (alpha) to track changing rewards
instead of the standard averaging method.
"""

import numpy as np
from Week7_3 import NonStationaryBandit


class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        """
        Initialize the agent
        k: number of arms
        epsilon: exploration probability
        alpha: constant step-size for updating estimated values
        """
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k)  # estimated value of each action

    def select_action(self):
        """Select action using epsilon-greedy strategy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.k)  # explore
        else:
            return np.argmax(self.Q)  # exploit

    def update(self, action, reward):
        """Update estimated value of the selected action"""
        self.Q[action] += self.alpha * (reward - self.Q[action])


# Simulation of 10,000 steps
if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    bandit = NonStationaryBandit()
    agent = EpsilonGreedyAgent(epsilon=0.1, alpha=0.1)

    rewards = []
    for t in range(10000):
        action = agent.select_action()
        reward = bandit.step(action)
        agent.update(action, reward)
        rewards.append(reward)

    print(f"Average reward over 10000 steps: {np.mean(rewards):.3f}")

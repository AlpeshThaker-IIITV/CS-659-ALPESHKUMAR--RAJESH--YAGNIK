"""
10-Armed Non-Stationary Bandit
--------------------------------
This script defines a 10-armed bandit where each arm's mean reward starts equal and
then undergoes an independent random walk at every time step. The reward returned
for an action is sampled from a normal distribution centered at the current mean of
that arm.
"""

import numpy as np


class NonStationaryBandit:
    def __init__(self, k=10, std_walk=0.01):
        """
        Initialize the 10-armed bandit.
        k: number of arms (default 10)
        std_walk: standard deviation of random walk for mean rewards
        """
        self.k = k
        self.std_walk = std_walk
        self.means = np.zeros(k)  # all arms start with mean reward = 0

    def step(self, action):
        """
        Take an action and return a reward.
        Action: integer from 0 to k-1
        """
        # Random reward sampled from current mean with small Gaussian noise
        reward = np.random.normal(self.means[action], 1.0)

        # Update each arm's mean via independent random walk
        self.means += np.random.normal(0, self.std_walk, self.k)

        return reward


# Example usage
if __name__ == "__main__":
    bandit = NonStationaryBandit()
    for t in range(5):
        action = np.random.randint(0, 10)  # pick a random arm
        reward = bandit.step(action)
        print(f"Step {t + 1}, Action: {action}, Reward: {reward:.2f}, Means: {bandit.means}")

import numpy as np
import matplotlib.pyplot as plt

# Define binary bandits
def binaryBanditA(action):
    p = [0.1, 0.2]
    return 1 if np.random.rand() < p[action-1] else 0

def binaryBanditB(action):
    p = [0.8, 0.9]
    return 1 if np.random.rand() < p[action-1] else 0

# Epsilon-greedy algorithm with tracking of actions
def epsilon_greedy(bandit_func, n_steps=1000, epsilon=0.1):
    Q = [0, 0]          # Estimated values
    N = [0, 0]          # Number of times each action is chosen
    rewards = []
    actions = []

    for step in range(n_steps):
        # Explore or exploit
        if np.random.rand() < epsilon:
            action = np.random.choice([1, 2])
        else:
            action = np.argmax(Q) + 1

        reward = bandit_func(action)
        rewards.append(reward)
        actions.append(action)

        # Update estimates
        N[action-1] += 1
        Q[action-1] += (reward - Q[action-1]) / N[action-1]

    return Q, rewards, actions

# Run for Bandit A
Q_A, rewards_A, actions_A = epsilon_greedy(binaryBanditA)
# Run for Bandit B
Q_B, rewards_B, actions_B = epsilon_greedy(binaryBanditB)

print("Estimated values for Bandit A:", Q_A)
print("Estimated values for Bandit B:", Q_B)
print("Average reward for Bandit A:", np.mean(rewards_A))
print("Average reward for Bandit B:", np.mean(rewards_B))

# Plot cumulative average reward
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(np.cumsum(rewards_A)/np.arange(1, len(rewards_A)+1), label='Bandit A')
plt.plot(np.cumsum(rewards_B)/np.arange(1, len(rewards_B)+1), label='Bandit B')
plt.xlabel('Steps')
plt.ylabel('Cumulative Average Reward')
plt.title('Learning Curve')
plt.legend()

# Plot action selection
plt.subplot(1,2,2)
plt.plot(np.cumsum([a==1 for a in actions_A])/np.arange(1,len(actions_A)+1), label='Action 1 - Bandit A')
plt.plot(np.cumsum([a==2 for a in actions_A])/np.arange(1,len(actions_A)+1), label='Action 2 - Bandit A')
plt.plot(np.cumsum([a==1 for a in actions_B])/np.arange(1,len(actions_B)+1), label='Action 1 - Bandit B')
plt.plot(np.cumsum([a==2 for a in actions_B])/np.arange(1,len(actions_B)+1), label='Action 2 - Bandit B')
plt.xlabel('Steps')
plt.ylabel('Fraction of Times Chosen')
plt.title('Action Selection Over Time')
plt.legend()
plt.tight_layout()
plt.show()

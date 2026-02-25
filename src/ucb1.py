import numpy as np

class UCB1:
    def __init__(self, arms):
        self.arms = arms
        self.N = len(arms)
        self.counts = np.zeros(self.N)
        self.values = np.zeros(self.N)
        self.t = 0

    def select_arm(self):
        self.t += 1
        for i in range(self.N):
            if self.counts[i] == 0:
                return i
        ucb_values = self.values + np.sqrt(2 * np.log(self.t) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = value + (reward - value) / n
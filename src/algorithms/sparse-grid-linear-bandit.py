import numpy as np
import itertools
import math

    
# 1. Synthetic Linear Bandit Environment
    

class LinearBanditEnv:
    """
    f(x) = theta^T x
    Reward = f(x) + Gaussian noise
    Domain: x in [0,1]^d
    """
    def __init__(self, d, noise_std=0.01):
        self.d = d
        self.theta = np.random.uniform(-1, 1, size=d)
        self.noise_std = noise_std

        # True continuous optimum over [0,1]^d
        # For linear function on cube: choose 1 where theta > 0, else 0
        self.x_star = (self.theta > 0).astype(float)
        self.f_star = self.theta @ self.x_star

    def reward(self, x):
        return self.theta @ x + np.random.normal(0, self.noise_std)


# 2. Sparse Grid (Simplified Smolyak-like construction)

def one_dimensional_nodes(level):
    if level == 1:
        return np.array([0.5])
    else:
        n = 2**(level-1)
        return np.linspace(0, 1, n+1)


def smolyak_grid(d, max_level):
    nodes = set()

    # All multi-indices
    for levels in itertools.product(range(1, max_level+1), repeat=d):
        if sum(levels) <= max_level + d - 1:
            grids_1d = [one_dimensional_nodes(l) for l in levels]
            for point in itertools.product(*grids_1d):
                nodes.add(tuple(point))

    return np.array(list(nodes))


 
# 3. UCB1 for discrete arms
    

class UCB1:
    def __init__(self, arms):
        self.arms = arms
        self.N = len(arms)
        self.counts = np.zeros(self.N)
        self.values = np.zeros(self.N)
        self.t = 0

    def select_arm(self):
        self.t += 1

        # Play each arm once initially
        for i in range(self.N):
            if self.counts[i] == 0:
                return i

        ucb_values = self.values + np.sqrt(2 * np.log(self.t) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]

        # Incremental mean update
        self.values[arm_index] = value + (reward - value) / n


    
# 4. Main experiment
    

def run_experiment(d=3, T=5000, max_level=3):
    env = LinearBanditEnv(d)

    # Sparse grid discretization
    grid = smolyak_grid(d, max_level)
    N = len(grid)

    print(f"Dimension: {d}")
    print(f"Number of sparse grid nodes: {N}")

    agent = UCB1(grid)

    cumulative_regret = 0
    regrets = []

    for t in range(T):
        arm_index = agent.select_arm()
        x = grid[arm_index]

        reward = env.reward(x)
        agent.update(arm_index, reward)

        # Regret computed w.r.t true continuous optimum
        regret = env.f_star - env.theta @ x
        cumulative_regret += regret
        regrets.append(cumulative_regret)

    return regrets


    
# 5. Run example
    

if __name__ == "__main__":
    d = 5
    T = 5000
    max_level = 3  # increase for finer grid

    regrets = run_experiment(d=d, T=T, max_level=max_level)

    import matplotlib.pyplot as plt
    plt.plot(regrets)
    plt.title("Cumulative Regret")
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.show()
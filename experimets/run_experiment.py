import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.sparse_grid import smolyak_grid
from src.ucb1 import UCB1
from experiments.environment import LinearBanditEnv

def run_experiment(d=5, T=5000, max_level=3):
    env = LinearBanditEnv(d)
    grid = smolyak_grid(d, max_level)
    agent = UCB1(grid)

    cumulative_regret = 0
    regrets = []

    for t in range(T):
        arm_index = agent.select_arm()
        x = grid[arm_index]
        reward = env.reward(x)
        agent.update(arm_index, reward)
        regret = env.f_star - env.theta @ x
        cumulative_regret += regret
        regrets.append(cumulative_regret)

    return regrets

if __name__ == "__main__":
    regrets = run_experiment(d=5, T=5000, max_level=3)

    os.makedirs("results/plots", exist_ok=True)
    plt.plot(regrets)
    plt.title("Cumulative Regret — Sparse Grid UCB1")
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.savefig("results/plots/cumulative_regret.png", dpi=150, bbox_inches='tight')
    plt.show()
```
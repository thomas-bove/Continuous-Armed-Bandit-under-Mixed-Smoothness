import numpy as np

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
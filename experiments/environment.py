# environment.py
import numpy as np
from scipy.optimize import differential_evolution

class LinearBanditEnv:
    def __init__(self, d, noise_std=0.01):
        self.d = d
        self.theta = np.random.uniform(-1, 1, size=d)
        self.noise_std = noise_std
        self.x_star = (self.theta > 0).astype(float)
        self.f_star = self.theta @ self.x_star

    def reward(self, x):
        return self.theta @ x + np.random.normal(0, self.noise_std)


class DMSBanditEnv:
    

    def __init__(self, d, noise_std=0.01):
     self.d = d
     self.noise_std = noise_std
     bounds = [(0, 1)] * d
     result = differential_evolution(
        lambda x: -self._f(x), 
        bounds, 
        seed=42, 
        tol=1e-6
     )
     self.x_star = result.x
     self.f_star = -result.fun

    def _f(self, x):
        return np.sum(np.sin(np.pi * x)) + np.prod(x)

    def reward(self, x):
        return self._f(x) + np.random.normal(0, self.noise_std)
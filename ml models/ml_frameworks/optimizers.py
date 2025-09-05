#Currently does not support cpu as I have mostly moved on to training on the GPU

import cupy as cp

class Optimizer:
    def __init__(self, lr = 0.001):
        self.lr = lr

    def update(self, params, grads):
        raise NotImplementedError

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, grads):
        return self.lr * grads

class MomentumSGD:
    def __init__(self, lr=0.01, beta = 0.90):
        self.lr = lr
        self.beta = beta
        self.velocities = {}

    def update(self, param_name, grads):
        # Initialize velocity if it doesn't exist
        if param_name not in self.velocities:
            self.velocities[param_name] = cp.zeros_like(grads)

        # Update velocity
        v = self.velocities[param_name]
        v = self.beta * v + (1 - self.beta) * grads
        self.velocities[param_name] = v

        # Return the parameter update
        return self.lr * v

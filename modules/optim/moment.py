"""
sgd optimizer with momentum

https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from modules.optim.base import Optimizer

class Momentum(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                try:
                    layer.pre_grads
                except Exception:
                    layer.pre_grads = {key:0 for key in layer.params.keys()}
        

                    

    def step(self):
        """
        Performs a single optimization step.
        """
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.pre_grads[key] = self.momentum * layer.pre_grads[key] + self.lr * layer.grads[key]
                    layer.params[key]-=layer.pre_grads[key]

    def zero_grad(self):
        """
        Clear gradients
        """
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.grads[key]=None
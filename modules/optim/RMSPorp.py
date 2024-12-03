"""
RMSProp

https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from modules.optim.base import Optimizer

class RMSPorp(Optimizer):
    def __init__(self, model, lr=0.01, beta=0.9,eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.eps = eps
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                try:
                    layer.pre_grads_scale
                except Exception:
                    layer.pre_grads_scale = {key:0 for key in layer.params.keys()}
        

                    

    def step(self):
        """
        Performs a single optimization step.
        """
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.pre_grads_scale[key] = self.beta * layer.pre_grads_scale[key] + (1 - self.beta) * layer.grads[key] ** 2
                    layer.params[key]-=self.lr*layer.grads[key]/(np.sqrt(layer.pre_grads_scale[key]+self.eps))

    def zero_grad(self):
        """
        Clear gradients
        """
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.grads[key]=None
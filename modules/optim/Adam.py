"""
Adam

https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from modules.optim.base import Optimizer

class Adam(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9,beta=0.9,eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.momentum = momentum
        self.count=0
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                try:
                    layer.pre_grads_scale
                except Exception:
                    layer.pre_grads_scale = {key:0 for key in layer.params.keys()}
                try:
                    layer.pre_grads
                except Exception:
                    layer.pre_grads = {key:0 for key in layer.params.keys()}
        

    def step(self):
        """
        Performs a single optimization step.
        """
        self.count+=1
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.pred_grads[key] = self.momentum * layer.pre_grads[key] + (1 - self.momentum) * layer.grads[key]
                    layer.pre_grads_scale[key] = self.beta * layer.pre_grads_scale[key] + (1 - self.beta) * layer.grads[key] ** 2
                    layer.pred_grads[key]/=(1-self.momentum**self.count)
                    layer.pre_grads_scale[key]/=(1-self.beta**self.count)
                    layer.params[key]-=self.lr*layer.pred_grads[key]/(np.sqrt(layer.pre_grads_scale[key]+self.eps))

    def zero_grad(self):
        """
        Clear gradients
        """
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.grads[key]=None
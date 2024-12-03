"""
Activation functions

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py
"""

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from modules.modules.base import Module


class Sigmoid(Module):
    """Applies the element-wise function:
    .. math::
    \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}

    Shape:
    - input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - output: :math:`(*)`, same shape as the input.
    """

    def __init__(self,name):
        super().__init__(name)
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        input_clipped = np.clip(input, -100, 100)
        output = 1 / (1 + np.exp(-input_clipped))

        self.output = output
        return output

    def backward(self, output_grad):
        """
        Input:
            - output_grad：(*)
            partial (loss function) / partial (output of this module)

        Return：
            - input_grad：(*)
            partial (loss function) / partial (input of this module)
        """

        def grad_forward(x: np.ndarray) -> np.ndarray:
            y = np.exp(-x) / (1 + np.exp(-x)) ** 2
            return y

        input_clipped = np.clip(self.input, -100, 100)
        input_grad = output_grad * grad_forward(input_clipped)
        return input_grad


class ReLU(Module):
    """Applies the element-wise function:
    .. math::
    \text{ReLU}(x) = max(x,0)

    Shape:
    - input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - output: :math:`(*)`, same shape as the input.
    """

    def __init__(self,name):
        super().__init__(name)
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        # set negative elements to 0
        output = np.maximum(0, input)

        self.output = output
        return output

    def backward(self, output_grad):
        """
        Input:
            - output_grad：(*)
            partial (loss function) / partial (output of this module)

        Return：
            - input_grad：(*)
            partial (loss function) / partial (input of this module)
        """

        input_grad = output_grad * (self.input > 0)
        return input_grad

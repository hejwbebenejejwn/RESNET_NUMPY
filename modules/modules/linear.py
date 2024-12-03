"""
Linear Layer

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
"""
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from modules.modules.base import Module


class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`Y = XW^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Parameters:
        W: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        b: the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """

    def __init__(self, in_features, out_features, bias=True,name=""):
        super().__init__(name=name)
        # input and output
        self.input = None
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # params
        self.params = {}
        k = 1 / in_features
        self.params["W"] = np.random.uniform(
            low=-np.sqrt(k), high=np.sqrt(k), size=(out_features, in_features)
        )
        if bias:
            self.params["b"] = np.random.uniform(
                low=-np.sqrt(k), high=np.sqrt(k), size=(out_features)
            )

        # grads of params
        self.grads = {"W": None, "b": None}

    def forward(self, input):
        self.input = input
        output = input.dot(self.params["W"].T)
        if self.bias:
            output += self.params["b"]

        return output

    def backward(self, output_grad):
        """
        Input:
            - output_grad：(*, H_{out})
            partial (loss function) / partial (output of this module)

        Return：
            - input_grad：(*, H_{in})
            partial (loss function) / partial (input of this module)
        """

        def expand(x: np.ndarray) -> np.ndarray:
            b = x.swapaxes(0, -1)
            return np.stack([b[i].flatten() for i in range(b.shape[0])], axis=1)

        outgrad_expand = expand(output_grad)
        input_expand = expand(self.input)
        self.grads["W"] = outgrad_expand.T.dot(input_expand)
        if self.bias:
            self.grads["b"] = outgrad_expand.sum(axis=0)
        input_grad = output_grad.dot(self.params["W"])
        assert self.grads["W"].shape == self.params["W"].shape
        if self.bias:
            assert self.grads["b"].shape == self.params["b"].shape
        assert input_grad.shape == self.input.shape

        return input_grad

class flatten(Module):
    def __init__(self,name):
        super().__init__(name)

    def forward(self, data: np.ndarray):
        """data:(N,C,H,W),
        output:(N,C*H*W)"""
        self._N, self._C, self._H, self._W = data.shape
        return data.reshape(data.shape[0], -1)

    def backward(self, output_grad: np.ndarray):
        output_grad = np.stack(
            [
                output_grad[:, i * (self._H * self._W) : (i + 1) * (self._H * self._W)]
                for i in range(self._C)
            ],
            axis=1,
        )
        output_grad = np.stack(
            [
                output_grad[:, :, i * self._W : (i + 1) * self._W]
                for i in range(self._H)
            ],
            axis=2,
        )
        return output_grad
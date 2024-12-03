"""
Conv2D

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
"""

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from modules.modules.base import Module


class Conv2d(Module):
    """Applies a 2D convolution over an inputs signal composed of several inputs
    planes.

    Args:
        in_channels (int): Number of channels in the inputs image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the inputs. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from inputs
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        name="",
    ):
        super().__init__(name)
        # inputs and output
        self.inputs = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W_valid = None

        # params
        self.params = {}
        self.padding = (
            self.padding
            if isinstance(self.padding, tuple)
            else (self.padding, self.padding)
        )
        self.stride = (
            self.stride
            if isinstance(self.stride, tuple)
            else (self.stride, self.stride)
        )
        self.kernel_size = (
            self.kernel_size
            if isinstance(self.kernel_size, tuple)
            else (self.kernel_size, self.kernel_size)
        )
        k = 1 / in_channels / self.kernel_size[0] / self.kernel_size[1]
        if bias:
            self.params["b"] = np.random.uniform(
                -np.sqrt(k), np.sqrt(k), (out_channels,)
            )
        self.params["W"] = np.random.uniform(
            -np.sqrt(k),
            np.sqrt(k),
            size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
        )

        # grads of params
        self.grads = {"b": None, "W": None}

    def forward(self, inputs):
        self.inputs = inputs

        if inputs.ndim == 3:
            inputs = inputs.reshape(1, inputs.shape[0], inputs.shape[1], inputs.shape[2])
        assert (
            inputs.ndim == 4
        ), "Only 3D and 4D inputs are supported, got {}D instead".format(inputs.ndim)

        def padding(x: np.ndarray, pad: tuple):
            return np.pad(
                x, pad_width=((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1]))
            )

        def unit(xi: np.ndarray, wj: np.ndarray, b: int, stride: tuple):
            """xi:(Cin,Hin,Win),
            wj:(Cin,KS[0],KS[1]),
            """
            hi, wi = xi.shape[1:]
            k0, k1 = wj.shape[1:]
            ho = int(np.floor((hi - k0) / stride[0]) + 1)
            wo = int(np.floor((wi - k1) / stride[1]) + 1)
            y = np.zeros((ho, wo))
            for i in range(ho):
                for j in range(wo):
                    temp = xi[
                        :,
                        i * stride[0] : i * stride[0] + k0,
                        j * stride[1] : j * stride[1] + k1,
                    ]
                    y[i, j] = np.sum(temp * wj)
            return y + b

        self.input_padded = padding(inputs, self.padding)
        W = self.params["W"]
        b = self.params["b"]
        if self.kernel_size[0] > self.input_padded.shape[2]:
            W = W[..., : self.input_padded.shape[2], :]
        if self.kernel_size[1] > self.input_padded.shape[3]:
            W = W[..., : self.input_padded.shape[3]]
        self.W_valid = W

        output = np.stack(
            [
                np.stack(
                    [
                        unit(
                            self.input_padded[i],
                            W[j],
                            b[j],
                            self.stride,
                        )
                        for j in range(W.shape[0])
                    ],
                    axis=0,
                )
                for i in range(self.input_padded.shape[0])
            ],
            axis=0,
        )
        if self.inputs.ndim == 3:
            output = output[0]
        return output

    def backward(self, output_grad):
        if output_grad.ndim == 3:
            output_grad = output_grad.reshape(1, *output_grad.shape)
        assert (
            output_grad.ndim == 4
        ), "Only 3D and 4D output gradients are supported, got {}D instead".format(
            output_grad.ndim
        )

        def wgrad_ij(output_grad: np.ndarray, inputs: np.ndarray, i: int, j: int):
            inputij = inputs[
                :,
                :,
                i : i
                + self.stride[0] * (output_grad.shape[2] - 1)
                + 1 : self.stride[0],
                j : j
                + self.stride[1] * (output_grad.shape[3] - 1)
                + 1 : self.stride[1],
            ]

            output_grad_flatten=output_grad.swapaxes(0,1).reshape((output_grad.shape[1],-1))
            inputij_flatten =  inputij.swapaxes(0,1).reshape((inputij.shape[1],-1))
            return output_grad_flatten.dot(inputij_flatten.T)

        def xgradcd(output_grad: np.ndarray, c: int, d: int):
            kernel_size=self.W_valid.shape[2:]
            padh=c+1-kernel_size[0]
            padw=d+1-kernel_size[1]
            padh=max(padh,0)
            padw=max(padw,0)
            W=np.pad(self.W_valid,((0,0),(0,0),(0,padh),(0,padw)),"constant",constant_values=0)
            W=W[...,c::-self.stride[0],d::-self.stride[1]]
            h_diff=output_grad.shape[2]-W.shape[2]
            w_diff=output_grad.shape[3]-W.shape[3]
            if h_diff<0:
                W=W[...,:output_grad.shape[2],:]
                h_diff=0
            if w_diff<0:
                W=W[...,:output_grad.shape[3]]
                w_diff=0
            W=np.pad(W,((0,0),(0,0),(0,h_diff),(0,w_diff)),"constant",constant_values=0)
            output_grad_flatten=output_grad.reshape((output_grad.shape[0],-1))
            W_flatten =  W.swapaxes(0,1).reshape((W.shape[1],-1))
            return output_grad_flatten.dot(W_flatten.T)

        self.grads["b"] = output_grad.sum(axis=0).sum(axis=1).sum(axis=1)
        W_valid_grad = np.stack(
            [
                np.stack(
                    [
                        wgrad_ij(
                            output_grad=output_grad, inputs=self.input_padded, i=i, j=j
                        )
                        for i in range(self.W_valid.shape[2])
                    ],
                    axis=2,
                )
                for j in range(self.W_valid.shape[3])
            ],
            axis=3,
        )
        hdiff = self.params["W"].shape[2] - self.W_valid.shape[2]
        wdiff = self.params["W"].shape[3] - self.W_valid.shape[3]
        self.grads["W"] = np.pad(
            W_valid_grad,
            ((0, 0), (0, 0), (0, hdiff), (0, wdiff)),
            mode="constant",
            constant_values=0,
        )
        input_grad = np.stack(
            [
                np.stack(
                    [
                        xgradcd(output_grad=output_grad, c=c, d=d)
                        for c in range(self.input_padded.shape[-2])
                    ],
                    axis=2,
                )
                for d in range(self.input_padded.shape[-1])
            ],
            axis=3,
        )
        if self.padding[0]!=0:
            input_grad=input_grad[...,self.padding[0]:-self.padding[0],:]
        if self.padding[1]!=0:
            input_grad=input_grad[...,self.padding[1]:-self.padding[1]]
        if self.inputs.ndim == 3:
            input_grad = input_grad[0]
        return input_grad


if __name__ == "__main__":
    lc = Conv2d(2, 4, 3, 1, 0)
    x = np.random.rand(2, 2, 2, 2)
    out = lc.forward(x)
    outgrad = np.random.rand(*out.shape)
    lc.backward(outgrad)

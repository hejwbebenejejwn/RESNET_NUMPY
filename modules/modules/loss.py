"""
loss functions

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/loss.py
"""
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from modules.modules.base import Loss


class BCELoss(Loss):
    r"""Creates a criterion that measures the Binary Cross Entropy between the target and
    the input probabilities:
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],
    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then
    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}
    Note that the targets :math:`y` should be numbers between 0 and 1.

    Notice that if :math:`x_n` is either 0 or 1, one of the log terms would be
    mathematically undefined in the above loss equation. We choose to set
    :math:`\log (0) = -\infty`, since :math:`\lim_{x\to 0} \log (x) = -\infty`.
    However, an infinite term in the loss equation is not desirable for several reasons.
    For one, if either :math:`y_n = 0` or :math:`(1 - y_n) = 0`, then we would be
    multiplying 0 with infinity. Secondly, if we have an infinite loss value, then
    we would also have an infinite term in our gradient, since
    :math:`\lim_{x\to 0} \frac{d}{dx} \log (x) = \infty`.
    This would make BCELoss's backward method nonlinear with respect to :math:`x_n`,
    and using it for things like linear regression would not be straight-forward.
    Our solution is that BCELoss clamps its log function outputs to be greater than
    or equal to -100. This way, we can always have a finite loss value and a linear
    backward method.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.
    """

    def __init__(self, model=None, reduction="mean"):
        self.input = None
        self.target = None

        self.model = model
        self.reduction = reduction

    def forward(self, input, target):
        self.input = input
        self.target = target

        a = np.exp(-100)
        loss = -target * np.log(input + a) - (1 - target) * np.log(a + 1 - input)
        if self.reduction == "mean":
            loss = np.mean(loss)
        elif self.reduction == "sum":
            loss = np.sum(loss)
        return loss

    def backward(self):
        a = np.exp(-100)
        input_grad = -self.target / (a + self.input) + (1 - self.target) / (
            a + 1 - self.input
        )
        self.model.backward(input_grad)


class CrossEntropyLoss(Loss):
    r"""This criterion computes the cross entropy loss between input logits
    and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain the unnormalized logits for each class (which do `not` need
    to be positive or sum to 1, in general).
    `input` has to be a Tensor of size :math:`(C)` for unbatched input,
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
    `K`-dimensional case. The last being useful for higher dimension inputs, such
    as computing cross entropy loss per-pixel for 2D images.

    The `target` that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to applying :class:`~torch.nn.LogSoftmax`
      on an input, followed by :class:`~torch.nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

    .. note::
        The performance of this criterion is generally better when `target` contains class
        indices, as this allows for optimized computation. Consider providing `target` as
        class probabilities only when a single class label per minibatch item is too restrictive.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.
          If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.
        - Output: If reduction is 'none', shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of K-dimensional loss, depending on the shape of the input. Otherwise, scalar.


        where:

        .. math::
            \begin{aligned}
                C ={} & \text{number of classes} \\
                N ={} & \text{batch size} \\
            \end{aligned}
    """

    def __init__(self, model=None, use_batch=True, reduction="mean") -> None:
        self.input = None
        self.target = None
        self.use_batch = use_batch
        self.model = model
        self.reduction = reduction

    def forward(self, input: np.ndarray, target: np.ndarray):
        """input should be of shape (N,C) or (C), target should be correspondingly of shape (N) or () or of same shape as input"""
        self.input = input
        self.target = target

        if not self.use_batch:
            input = input.reshape(1, *input.shape)
            target = target.reshape(1, *target.shape)

        self.aggregate = (np.exp(input)).sum(axis=1)
        if target.ndim == input.ndim - 1:
            self.targettype = "class"
            select = np.zeros(input.shape[0])
            for i in range(input.shape[0]):
                select[i] = input[i, target[i]]
            loss = np.log(self.aggregate) - select
            if not self.use_batch:
                loss = loss[0]
            if self.reduction == "mean":
                loss = np.mean(loss)
            if self.reduction == "sum":
                loss = np.sum(loss)
        elif target.ndim == input.ndim:
            self.targettype = "probability"
            loss = target.sum(axis=1) * np.log(self.aggregate) - np.sum(
                target * input, axis=1
            )
            if not self.use_batch:
                loss = loss[0]
            if self.reduction == "mean":
                loss = np.mean(loss)
            if self.reduction == "sum":
                loss = np.sum(loss)


        return loss

    def backward(self):
        input = (
            self.input if self.use_batch else self.input.reshape(1, *self.input.shape)
        )
        target = (
            self.target
            if self.use_batch
            else self.target.reshape(1, *self.target.shape)
        )
        input_grad = np.zeros_like(input)
        if self.targettype == "class":
            for c in range(input.shape[1]):
                input_grad[:, c] = np.exp(input[:, c]) / self.aggregate - (target == c)
            if self.reduction == "mean":
                input_grad = input_grad / target.size
            if not self.use_batch:
                input_grad = input_grad[0]
        if self.targettype == "probability":
            input_grad = (
                np.exp(input).T * (target.sum(axis=1) / self.aggregate)
            ).T - target
            if self.reduction == "mean":
                input_grad = input_grad / target.size
            if not self.use_batch:
                input_grad = input_grad[0]

        self.model.backward(input_grad)


if __name__ == "__main__":
    c = CrossEntropyLoss()
    cs = CrossEntropyLoss(reduction="sum")
    abatch = np.random.rand(2, 3)
    ytrue = np.array([1, 2])
    print(c.forward(abatch, ytrue).shape)
    print(cs.forward(abatch, ytrue).shape)
    print(c.backward().shape)
    print(cs.backward().shape)

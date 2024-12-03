"""
base class
"""

class Module:
    """ base class of modules """
    def __init__(self,name):
        self.name = name
        self.params=None
        self.grads=None

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError


class Loss:
    """ base class of losses """

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
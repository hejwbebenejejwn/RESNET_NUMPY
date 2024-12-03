import modules.modules.linear as linear
import modules.modules.conv as conv
import modules.modules.pooling as pooling
import modules.modules.activation as activation
from modules.modules.batchnorm import BatchNorm2d
from modules.modules.base import Module
import numpy as np

class Downsample(Module):
    def __init__(self, in_channels, out_channels, stride=2,name=""):
        super().__init__(name)
        self.conv = conv.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True,name=self.name+"_conv")
        self.bn = BatchNorm2d(out_channels,name=self.name+"_bn")
        self.sublayers = [self.conv, self.bn]

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def backward(self, output_grad):
        grad=self.bn.backward(output_grad)
        grad=self.conv.backward(grad)
        return grad
    
    def train(self):
        self.bn.training=True
    
    def eval(self):
        self.bn.training=False

class BasicBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,name=""):
        super().__init__(name)
        self.conv1 = conv.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True,name=self.name+"_conv1")
        self.bn1 =BatchNorm2d(out_channels,name=self.name+"_bn1")
        self.relu1 = activation.ReLU(name=self.name+"_relu1")
        self.conv2 = conv.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,name=self.name+"_conv2")
        self.bn2 = BatchNorm2d(out_channels,name=self.name+"_bn2")
        self.relu2 = activation.ReLU(name=self.name+"_relu2")
        self.downsample = downsample
        self.stride = stride
        self.sublayers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2]
        if self.downsample is not None:
            self.sublayers.extend(self.downsample.sublayers)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out
    
    def backward(self, output_grad):
        output_grad=self.relu2.backward(output_grad)
        grad=self.bn2.backward(output_grad)
        grad=self.conv2.backward(grad)
        grad=self.relu1.backward(grad)
        grad=self.bn1.backward(grad)
        grad=self.conv1.backward(grad)
        if self.downsample is not None:
            grad+=self.downsample.backward(output_grad)
        else:
            grad+=output_grad
        return grad
    
    def train(self):
        self.bn1.training=True
        self.bn2.training=True
        if self.downsample is not None:
            self.downsample.train()

    def eval(self):
        self.bn1.training=False
        self.bn2.training=False
        if self.downsample is not None:
            self.downsample.eval()
        

class ResNet(Module):
    def __init__(self, name="resnet"):
        super().__init__(name)
        self.conv1 = conv.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True,name=self.name+"_conv1")
        self.bn1 = BatchNorm2d(64,name=self.name+"_bn1")
        self.relu = activation.ReLU(name=self.name+"_relu")
        self.maxpool = pooling.MaxPool2d(kernel_size=3, stride=2, padding=1,name=self.name+"_maxpool")
        self.layer1=self._resblock(64, 64, 2,name=self.name+"_layer1")
        self.layer2=self._resblock(64, 128, 2,2,name=self.name+"_layer2")
        self.layer3=self._resblock(128, 256, 2,2,name=self.name+"_layer3")
        self.layer4=self._resblock(256, 512, 2,2,name=self.name+"_layer4")
        self.avgpool = pooling.AvgPool2d(kernel_size=1, stride=1,padding=0,name=self.name+"_avgpool") # kernel_size = 1 for cifar10
        self.flatten=linear.flatten(name=self.name+"_flatten")
        self.fc=linear.Linear(512,10,name=self.name+"_fc")
        self.layers=[self.conv1, self.bn1, self.relu, self.maxpool]
        for block in self.layer1:
            self.layers.extend(block.sublayers)
        for block in self.layer2:
            self.layers.extend(block.sublayers)
        for block in self.layer3:
            self.layers.extend(block.sublayers)
        for block in self.layer4:
            self.layers.extend(block.sublayers)
        self.layers.extend([self.avgpool, self.flatten, self.fc])


    def _resblock(self, in_channels, out_channels, blocks,stride=1,name=""):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = Downsample(in_channels, out_channels, stride,name=name+"_downsample")
        layers = [BasicBlock(in_channels, out_channels, stride, downsample,name=name+"_block0")]
        for i in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels,name=name+f"_block{i}"))
        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def backward(self, output_grad):
        output_grad=self.fc.backward(output_grad)
        grad=self.flatten.backward(output_grad)
        grad=self.avgpool.backward(grad)
        for block in reversed(self.layer4):
            grad=block.backward(grad)
        for block in reversed(self.layer3):
            grad=block.backward(grad)
        for block in reversed(self.layer2):
            grad=block.backward(grad)
        for block in reversed(self.layer1):
            grad=block.backward(grad)
        grad=self.maxpool.backward(grad)
        grad=self.relu.backward(grad)
        grad=self.bn1.backward(grad)
        grad=self.conv1.backward(grad)
        return grad
    
    def train(self):
        self.bn1.training=True
        for block in self.layer1:
            block.train()
        for block in self.layer2:
            block.train()
        for block in self.layer3:
            block.train()
        for block in self.layer4:
            block.train()
        
    def eval(self):
        self.bn1.training=False
        for block in self.layer1:
            block.eval()
        for block in self.layer2:
            block.eval()
        for block in self.layer3:
            block.eval()
        for block in self.layer4:
            block.eval()

if __name__ == '__main__':
    block=BasicBlock(512,512,name="block")
    x=np.random.rand(5,512,1,1)
    y=block(x)
    block.backward(y)
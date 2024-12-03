from modules.modules.ResNet import ResNet
from modules.modules import loss
from modules.optim import sgd
from modules import trainer
from modules.metrics import accuracy

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

datapath=""
checkdir=""

class ToNumpyArray:
    def __call__(self, sample:torch.Tensor):
        # Convert tensor image to numpy array (C x H x W)
        return sample.numpy()


transform=transforms.Compose([
    transforms.ToTensor(),
    ToNumpyArray()
])

trainset = torchvision.datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform)

train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


model = ResNet()
metric = accuracy.Accuracy()
lossfunction = loss.CrossEntropyLoss(model)
trainor = trainer.Trainer(model, sgd.SGD, metric, lossfunction)
trainor.train(
    trainloader,
    valloader,
    num_epochs=300,
    log_epochs=5,
    save_dir=checkdir,
)
trainor.load_model(checkdir)

score,_=trainor.evaluate(testloader)
print(f'the accuracy on test set is {score}')
from modules.modules.base import Module
import numpy as np

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,name=""):
        super(BatchNorm2d, self).__init__(name)
        self.input=None
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = np.ones((num_features,1,1))  # scale parameter
        self.beta = np.zeros((num_features,1,1))  # shift parameter
        # running mean and variance for inference
        self.running_mean = np.zeros((num_features,1,1))
        self.running_var = np.ones((num_features,1,1))
        self.params={"gamma":self.gamma,"beta":self.beta}
        self.unlearnable_params={"running_mean":self.running_mean,"running_var":self.running_var}
        self.grads={"gamma":None,"beta":None}

    def forward(self, x:np.ndarray):
        self.input=x
        if self.training:
            batch_mean = x.mean(axis=0, keepdims=True)
            batch_var = x.var(axis=0, keepdims=True)
            
            x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * batch_var + (1 - self.momentum) * self.running_var
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        out = self.gamma * x_hat + self.beta
        return out

    def backward(self, output_grad):
        self.grads['beta']=output_grad
        gamma_grad=np.zeros_like(self.gamma)
        for i in range(self.num_features):
            gamma_grad[i,0,0]=(output_grad[:,i,...]*self.input[:,i,...]).sum()
        self.grads['gamma']=gamma_grad
        return output_grad*self.gamma

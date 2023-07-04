"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
import torch
from torch import nn
from torch.nn.parameter import Parameter # import Parameter for custom activations

class SharpActivation(nn.Module):
    def __init__(self, in_features,beta = None, gamma = None):
        super(SharpActivation,self).__init__()
        self.in_features = in_features
        
        if beta == None:
            self.beta = Parameter(torch.zeros(self.in_features)) # create a tensor out of alpha
        else:
            self.beta = Parameter(torch.tensor(beta)) # create a tensor out of alpha
        
        if gamma == None:
            self.gamma = Parameter(torch.zeros(self.in_features)) # create a tensor out of alpha
        else:
            self.gamma = Parameter(torch.tensor(gamma)) # create a tensor out of alpha
        
    def forward(self,x):
        a = self.gamma + ((1+torch.exp(-torch.mul(self.beta,x)))**-1)*(1-self.gamma)
        result = torch.mul(a,x)
        return result

class NeuralNetwork(nn.Module):
    """
    A class that determines a neural network structure
    
    Attributes
    ----------
    flatten : method
        nn.flatten() from pytorch
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    double : method
        converts all parameters to doubles rather than float
    """
    def __init__(self,num_pars,data_len):
        super().__init__()
        self.p = 0.2
        self.LinearStack1 = nn.Sequential(
            nn.Linear(num_pars,256),
            nn.ReLU()
            ) 
        self.LinearStack2 = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU())
        self.LinearStack3 = nn.Sequential(
            nn.Linear(512,data_len))
        self.dropout1 = nn.Dropout(p=self.p)
        self.dropout2 = nn.Dropout(p=self.p)
        self.double()
    
    def forward(self,pars):
        stack1 = self.LinearStack1(pars)
        stack1 = self.dropout1(stack1)
        stack2 = self.LinearStack2(stack1)
        stack2 = self.dropout2(stack2)
        result = self.LinearStack3(stack2)
        return result

class SharpNetwork(NeuralNetwork):
    """
    A class that determines a variant of the neural network architecture
    that uses a custom activation function
    
    Attributes
    ----------
    flatten : method
        nn.flatten() from pytorch
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    double : method
        converts all parameters to doubles rather than float
    """
    
    def __init__(self,num_pars,data_len):
        super(SharpNetwork,self).__init__()
        self.p = 0.2
        self.LinearStack1 = nn.Sequential(
            nn.Linear(num_pars,256),
            SharpActivation(256)
            ) 
        self.LinearStack2 = nn.Sequential(
            nn.Linear(256,512),
            SharpActivation(512))
        self.LinearStack3 = nn.Sequential(
            nn.Linear(512,data_len))
        self.dropout1 = nn.Dropout(p=self.p)
        self.dropout2 = nn.Dropout(p=self.p)
        self.double()

class Committee(NeuralNetwork):
    """
    A class that determines a variant of the neural network infrastructure
    that adds a dropout to create a committee format when performing active
    learning.
    
    Attributes
    ----------
    flatten : method
        nn.flatten() from pytorch
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    double : method
        converts all parameters to doubles rather than float
    """
    
    def forward(self,pars):
        stack1 = self.LinearStack1(pars)
        stack2 = self.dropout1(stack1)
        stack3 = self.LinearStack2(stack2)
        stack4 = self.dropout2(stack3)
        stack5 = self.LinearStack3(stack4)
        stack6 = self.dropout3(stack5)
        result = self.LinearStack3(stack6)
        return result
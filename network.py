"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
from torch import nn

class Committee(nn.Module):
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
    
    def __init__(self,num_pars,data_len):
        super().__init__()
        self.p = 0.5
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
        stack2 = self.dropout1(stack1)
        stack3 = self.LinearStack2(stack2)
        stack4 = self.dropout2(stack3)
        result = self.LinearStack3(stack4)
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
        self.p = 0.5
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
        stack2 = self.LinearStack2(stack1)
        result = self.LinearStack3(stack2)
        return result
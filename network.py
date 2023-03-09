"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
from torch import nn


class NeuralNetwork(nn.Module):
    
    def __init__(self,data_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.LinearStack = nn.Sequential(
            nn.Linear(26,data_len)
            ) 
        self.double()
    
    def forward(self,pars):
        pars = self.flatten(pars)
        result = self.LinearStack(pars)
        return result


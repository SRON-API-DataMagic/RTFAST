"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
from torch import nn


class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.LinearStack = nn.Sequential(
            nn.linear(25,4096),
            nn.Relu()
            ) 
    
    def forward(self,pars):
        result = self.LinearStack(pars)
        return result


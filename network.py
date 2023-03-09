"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
from torch import nn


class NeuralNetwork(nn.Module):
    
    def __init__(self,data_len):
        super().__init__()
        self.LinearStack = nn.Sequential(
            nn.Linear(None,data_len),
            nn.ReLU()
            ) 
    
    def forward(self,pars):
        result = self.LinearStack(pars)
        return result


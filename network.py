"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
from torch import nn

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
    
    def __init__(self,data_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.LinearStack = nn.Sequential(
            nn.Linear(2,256),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(256,512),
            nn.Dropout(),
            nn.Softplus(),
            nn.Linear(512,data_len)
            ) 
        self.double()
    
    def forward(self,pars):
        result = self.LinearStack(pars)
        return result


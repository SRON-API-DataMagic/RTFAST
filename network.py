"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
import torch
from torch import nn
from torch.nn.parameter import Parameter # import Parameter for custom activations

class SharpActivation(nn.Module):
    """
    Variant activation function that has trainable parameters to focus parts
    of a neural network on different functions. Particularly effective at sharp
    gradients.
    """
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
    p : float
        determines dropout probability of the dropout layers
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    dropout : method
        Randomly zeros nodes in the network to emulate ensemble training
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
    p : float
        determines dropout probability of the dropout layers
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    dropout : method
        Randomly zeros nodes in the network to emulate ensemble training
    double : method
        converts all parameters to doubles rather than float
    """
    
    def __init__(self,num_pars,data_len):
        super(SharpNetwork,self).__init__(num_pars,data_len)
        self.p = 0.2
        self.LinearStack1 = nn.Sequential(
            nn.Linear(num_pars,256),
            SharpActivation(256)
            ) 
        self.LinearStack2 = nn.Sequential(
            nn.Linear(256,512),
            SharpActivation(512))
        self.LinearStack3 = nn.Sequential(
            nn.Linear(512,1024),
            SharpActivation(1024))
        self.LinearStack4 = nn.Sequential(
            nn.Linear(1024,data_len))
        self.dropout1 = nn.Dropout(p=self.p)
        self.dropout2 = nn.Dropout(p=self.p)
        self.dropout3 = nn.Dropout(p=self.p)
        self.double()
    
    def forward(self,pars):
        stack1 = self.LinearStack1(pars)
        stack1 = self.dropout1(stack1)
        stack2 = self.LinearStack2(stack1)
        stack2 = self.dropout2(stack2)
        stack3 = self.LinearStack3(stack2)
        stack3 = self.dropout3(stack3)
        result = self.LinearStack4(stack3)
        return result

class HeavyFluxNetwork(SharpNetwork):
    """
    A variant network of SharpNetwork that features one extra hidden layer.
    
    Attributes
    ----------
    p : float
        determines dropout probability of the dropout layers
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    dropout : method
        Randomly zeros nodes in the network to emulate ensemble training
    double : method
        converts all parameters to doubles rather than float
    """
    def __init__(self,num_pars,data_len):
        super().__init__(num_pars,data_len)
        self.LinearStack4 = nn.Sequential(
            nn.Linear(1024,2048),
            SharpActivation(2048))
        self.LinearStack5 = nn.Sequential(
            nn.Linear(2048,data_len))
        self.dropout4 = nn.Dropout(p=self.p)
        self.double()
    
    def forward(self,pars):
        stack1 = self.LinearStack1(pars)
        stack1 = self.dropout1(stack1)
        stack2 = self.LinearStack2(stack1)
        stack2 = self.dropout2(stack2)
        stack3 = self.LinearStack3(stack2)
        stack3 = self.dropout3(stack3)
        stack4 = self.LinearStack4(stack3)
        stack4 = self.dropout4(stack4)
        result = self.LinearStack5(stack4)
        return result
    
class LagsNetwork(nn.Module):
    """
    A class that determines the neural network architecture for learning the 
    time lags output of rtdist. This class features a distinct difference to
    the flux version of this network: it produces sigmoid values for the
    likelihood that a given value for an energy bin is negative or positive.
    This allows us to avoid issues with time lags spanning many orders of
    magnitude in both negative and positive values.
    
    Attributes
    ----------
    p : float
        determines dropout probability of the dropout layers
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    OutputSigmoid: method
        Places a sigmoid on output values of whether a given energy bin is
        positive or negative.
    dropout : method
        Randomly zeros nodes in the network to emulate ensemble training
    double : method
        converts all parameters to doubles rather than float
    """
    
    def __init__(self,num_pars,data_len):
        super().__init__()
        self.p = 0.2
        self.LinearStack1 = nn.Sequential(
            nn.Linear(num_pars,256),
            SharpActivation(256)
            ) 
        self.LinearStack2 = nn.Sequential(
            nn.Linear(256,512),
            SharpActivation(512))
        self.LinearStack3 = nn.Sequential(
            nn.Linear(512,1024),
            SharpActivation(1024))
        self.OutputAbsolute = nn.Sequential(
            nn.Linear(1024,data_len))
        self.OutputIndex = nn.Sequential(
            nn.Linear(1024,data_len))
        self.OutputSigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=self.p)
        self.dropout2 = nn.Dropout(p=self.p)
        self.dropout3 = nn.Dropout(p=self.p)
        self.double()
        
    def forward(self,pars):
        stack1 = self.LinearStack1(pars)
        stack1 = self.dropout1(stack1)
        stack2 = self.LinearStack2(stack1)
        stack2 = self.dropout2(stack2)
        stack3 = self.LinearStack3(stack2)
        stack3 = self.dropout3(stack3)
        result = self.OutputAbsolute(stack3)
        ind = self.OutputSigmoid(self.OutputIndex(stack3))
        return result, ind

class HeavyLagsNetwork(LagsNetwork):
    """
    A variant of the LagsNetwork class that adds one extra hidden layer.
    
    Attributes
    ----------
    p : float
        determines dropout probability of the dropout layers
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    OutputSigmoid: method
        Places a sigmoid on output values of whether a given energy bin is
        positive or negative.
    dropout : method
        Randomly zeros nodes in the network to emulate ensemble training
    double : method
        converts all parameters to doubles rather than float
    """
    
    def __init__(self,num_pars,data_len):
        super().__init__(num_pars,data_len)
        self.LinearStack4 = nn.Sequential(
            nn.Linear(1024,2048),
            SharpActivation(2048))
        self.OutputAbsolute = nn.Sequential(
            nn.Linear(2048,data_len))
        self.OutputIndex = nn.Sequential(
            nn.Linear(2048,data_len))
        self.dropout4 = nn.Dropout(p=self.p)
        self.double()
    
    def forward(self,pars):
        stack1 = self.LinearStack1(pars)
        stack1 = self.dropout1(stack1)
        stack2 = self.LinearStack2(stack1)
        stack2 = self.dropout2(stack2)
        stack3 = self.LinearStack3(stack2)
        stack3 = self.dropout3(stack3)
        stack4 = self.LinearStack4(stack3)
        stack4 = self.dropout4(stack4)
        result = self.OutputAbsolute(stack4)
        ind = self.OutputSigmoid(self.OutputIndex(stack4))
        return result, ind

class PCAFluxNetwork(nn.Module):
    """
    Neural network class that aims to learn the relationship between the PCA
    components describing the flux spectrum and the original parameters. Made
    to be light-weight due to the small amount of input and output components.
    """
    
    def __init__(self,num_pars,output_len):
        super().__init__()
        self.LinearStack = nn.Sequential(nn.Linear(num_pars, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(), 
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(), 
                                         nn.Linear(512, output_len))
    
    def forward(self,pars):
        return self.LinearStack(pars)

class PCANetwork(nn.Module):
    """
    Neural network class that aims to learn the relationship between the PCA
    components describing the flux spectrum and the original parameters. Made
    to be light-weight due to the small amount of input and output components.
    """
    
    def __init__(self,num_pars,output_len):
        super().__init__()
        self.LinearStack = nn.Sequential(nn.Linear(num_pars, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(), 
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Linear(512, 512),
                                         nn.GELU(), 
                                         nn.Linear(512, output_len))
    
    def forward(self,pars):
        return self.LinearStack(pars)
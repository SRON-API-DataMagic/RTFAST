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

class LagsNetwork(nn.Module):
    
    def __init__(self,num_pars,data_len):
        super().__init__()
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
        ind = torch.round(self.OutputSigmoid(self.OutputIndex(stack3)))
        return result, ind
    
class LightSharpNetwork(SharpNetwork):
    
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

class Committee(NeuralNetwork):
    """
    A class that determines a variant of the neural network infrastructure
    that adds a dropout to create a committee format when performing active
    learning.
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
    
class DeepNetwork(nn.Module):
    """
    A class that determines a variant neural network structure featuring deeper
    but narrower layers
    
    Attributes
    ----------
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    HiddenStack : Sequential neural network layers
        Feedforward neural network callable in one method
    OutputStack : Sequential neural network layers
        Feedforward neural network callable in one method
    double : method
        converts all parameters to doubles rather than float
    """
    
    def __init__(self,num_pars,data_len):
        super().__init__()
        self.p = 0.2
        self.LinearStack = nn.Sequential(
            nn.Linear(num_pars, 100),
            SharpActivation(100),
            nn.Dropout(self.p)
            )
        self.HiddenStack = nn.Sequential(
            nn.Linear(100,100),
            SharpActivation(100),
            nn.Dropout(self.p),
            nn.Linear(100,100),
            SharpActivation(100),
            nn.Dropout(self.p),
            nn.Linear(100,100),
            SharpActivation(100),
            nn.Dropout(self.p),
            nn.Linear(100,data_len))
        self.double()
    
    def forward(self,pars):
        stack1 = self.LinearStack(pars)
        results = self.HiddenStack(stack1)
        return results

class DeepResNetwork(nn.Module):
    """
    A class that is a variant of the deep neural network that utilises residual
    strategies in passing data through the network by adding the output of a
    previous stack to the output of the new stack.
    
    Attributes
    ----------
    LinearStack : Sequential neural network layers
        Feedforward neural network callable in one method
    HiddenStack : Sequential neural network layers
        Feedforward neural network callable in one method
    OutputStack : Sequential neural network layers
        Feedforward neural network callable in one method
    double : method
        converts all parameters to doubles rather than float
    """
    
    def __init__(self,num_pars,data_len):
        super().__init__()
        self.p = 0.2
        self.LinearStack = nn.Sequential(
            nn.Linear(num_pars, 100),
            SharpActivation(100),
            nn.Dropout(self.p)
            )
        self.HiddenStack1 = nn.Sequential(
            nn.Linear(100,100),
            SharpActivation(100),
            nn.Dropout(self.p))
        self.HiddenStack2 = nn.Sequential(
            nn.Linear(100,100),
            SharpActivation(100),
            nn.Dropout(self.p))
        self.HiddenStack3 = nn.Sequential(
            nn.Linear(100,100),
            SharpActivation(100),
            nn.Dropout(self.p))
        self.OutputStack = nn.Sequential(
            nn.Linear(100,data_len))
        
    def forward(self,pars):
        stack1 = self.LinearStack(pars)
        stack2 = self.HiddenStack1(stack1) + stack1
        stack3 = self.HiddenStack2(stack2) + stack2
        stack4 = self.HiddenStack3(stack3) + stack3
        result = self.OutputStack(stack4)
        return result
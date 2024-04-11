"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
import torch
from torch import nn
from torch.nn.parameter import Parameter #needed for custom activation functions

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

class rtdist_ta_spec_emu(nn.Module):
    """
    Final neural network emulator architecture. Translates parameters into
    rtdist's time averaged spectrum output. Distinct from the cross-spectrum
    emulator.
    
    Composed of 8 hidden layers, each with 256 nodes. Must be paired with the
    standard scalers and PCA trained with the network to output rtdist
    values directly.
    """
    
    def __init__(self):
        super().__init__()
        self.LinearStack = nn.Sequential(nn.Linear(20, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 256),
                                         nn.GELU(),
                                         nn.Linear(256, 36))
        
        def forward(self,pars):
            return self.LinearStack(pars)
    
class DynamicNetwork(nn.Module):
    """
    Neural network used in hyperparameter sweeps. The number of layers and
    number of nodes in each layer can be specified at initialisation. It is
    recommended that any DynamicNetworks that are fully trained have their
    own fixed class written after a best model is found for the ease of the
    final user.
    """
    
    def __init__(self,num_pars,output_len,num_layers,nodes,activation):
        super().__init__()
        if activation == "GELU":
            act_type = nn.GELU()
        modules = []
        #specify input stack
        modules.append(nn.Linear(num_pars, nodes))
        modules.append(act_type)
        #dynamically add layers
        for i in range(num_layers):
            modules.append(nn.Linear(nodes, nodes))
            modules.append(act_type)
        #add output stack
        modules.append(nn.Linear(nodes, output_len))
        self.LinearStack = nn.Sequential(*modules)
        
    def forward(self,pars):
        return self.LinearStack(pars)
        
class DynamicCrossNetwork(nn.Module):
    """
    Neural network used in hyperparameter sweeps. THe number of layers and 
    number of nodes in said layers can be specified at initialisation. It is 
    recommended that any DynamicCrossNetworks that are fully trained have their
    own fixed class written after a best model is found for the ease of the
    final layer.
    
    Distinct from DynamicNetwork, this class features a final layer that
    produces a 2 dimensional array as the final output - corresponding to the
    real and imaginary parts of the cross-spectrum.
    """
    
    def __init__(self,num_pars,output_len,num_layers,nodes,activation):
        super().__init__()
        if activation == "GELU":
            act_type = nn.GELU()
        modules = []
        #specify input stack
        modules.append(nn.Linear(num_pars, nodes))
        modules.append(act_type)
        #dynamically add layers
        for i in range(num_layers):
            modules.append(nn.Linear(nodes, nodes))
            modules.append(act_type)
        #add output stack
        modules.append(nn.Linear(nodes, output_len))
        self.LinearStack = nn.Sequential(*modules)
        self.OutputStack = nn.Linear(1,2)
        
    def forward(self,pars):
        pred = self.LinearStack(pars)
        #next line reshapes so linear layer can split into real and imaginary
        #parts
        reshaped_pred = torch.reshape(pred,(pred.shape[0],pred.shape[1],1))
        return self.OutputStack(reshaped_pred)
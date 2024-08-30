"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
import torch
from torch import nn, vmap
from torch.func import stack_module_state, functional_call
import copy
from joblib import load
import numpy as np

class RtdistSpec(nn.Module):
    """
    Final neural network emulator architecture. Translates parameters into
    rtdist's time averaged spectrum output. Distinct from the cross-spectrum
    emulator.
    
    Composed of 8 hidden layers, each with 256 nodes. Must be paired with the
    standard scalers and PCA trained with the network to output rtdist
    values directly.
    """
    
    def __init__(self,pars=17,comps=40):
        super().__init__()
        self.LinearStack = nn.Sequential(nn.Linear(pars, 256),
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
                                         nn.Linear(256, comps))
        
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
    
    def __init__(self,num_pars,output_len,num_layers,nodes,activation="GELU"):
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

class DynamicResNetwork(nn.Module):
    """
    Neural network used in hyperparameter sweeps. The number of layers and
    number of nodes in each layer can be specified at initialisation. It is
    recommended that any DynamicNetworks that are fully trained have their
    own fixed class written after a best model is found for the ease of the
    final user.
    """
    def __init__(self, num_pars,output_len,num_residual_blocks=12, nodes = 256):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(num_pars, nodes),
            nn.ReLU()
        )
        
        # Create a list of residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nodes, nodes),
                nn.BatchNorm1d(nodes)
            ) for _ in range(num_residual_blocks)
        ])
        self.activation = nn.ReLU()
        
        self.output = nn.Sequential(
            nn.Linear(nodes, output_len)
        )

    def forward(self, x):
        x = self.input(x)
        for block in self.residual_blocks:
            x = x + block(x)
            x = self.activation(x)
        pred = self.output(x)
        return pred

class RTFAST_single(nn.Module):
    """
    This can be called to utilise the emulator automatically and output only
    spectra. This will automatically load in scalers and PCA objects required
    for computation. Instrumental effects are not included.
    
    Input a set of parameters and retrieve the spectrum.
    """
    def __init__(self,device=torch.device('cpu')):
        super().__init__()
        self.core = RtdistSpec()
        self.core.load_state_dict(torch.load("models/20_pars_flux.pth",
                                             map_location=device))
        self.core.eval() #turns off any training type layers
        self.core.double() #sets all parameters to double type
        
        self.pca  = load("scalers/PCA_20_spec.bin")
        self.comp = load("scalers/comp_20_spec.bin")
        self.spec = load("scalers/spec_20_spec.bin")
        self.calib = np.loadtxt("scalers/calib_factor.txt")
        
        self.pca_mean       = torch.Tensor(self.pca.mean_).double()
        self.pca_components = torch.Tensor(self.pca.components_).double()
        self.comp_mean      = torch.Tensor(self.comp.mean_).double()
        self.comp_scale     = torch.Tensor(self.comp.scale_).double()
        self.spec_mean      = torch.Tensor(self.spec.mean_).double()
        self.spec_scale     = torch.Tensor(self.spec.scale_).double()
        self.calib          = torch.Tensor(self.calib).double()
        
        self.powers         = [0,2,3,4,7,8,10,11,12,13,19]
    
    def PCA_inverse_transform(self,data_reduced):
        pca_comps = torch.matmul(data_reduced, self.pca_components) + self.pca_mean
        return pca_comps
    
    def comp_inverse_transform(self,data_reduced):
        components = torch.mul(data_reduced,self.comp_scale) + self.comp_mean
        return components
    
    def spec_inverse_transform(self,data_reduced):
        spectra = torch.mul(data_reduced,self.spec_scale) + self.spec_mean
        return spectra
    
    def pars_shift(self,theta):
        #adapts parameters to correct shape of 
        if theta.shape[0] == 20:
            theta[self.powers] = torch.log10(theta[self.powers])
        else:
            theta[:,self.powers] = torch.log10(theta[:,self.powers])
        return theta
    
    def forward(self,theta):
        theta = self.pars_shift(theta)
        data = self.core(theta)
        PCA_comps = self.comp_inverse_transform(data)
        std_spec = self.PCA_inverse_transform(PCA_comps)
        spectrum = 10**self.spec_inverse_transform(std_spec)
        spectrum = spectrum/self.calib
        return spectrum
        
class RtdistSpec_ensemble(nn.Module):
    def __init__(self,device=torch.device('cpu'),num_models=10):
        super().__init__()
        models = [DynamicNetwork(17,200,12,256).to(device) for _ in range(num_models)]
        for i,model in enumerate(models):
            model.load_state_dict(torch.load(f"models/{i}_20_pars_flux.pth",
                                                 map_location=device))
        
        self.ensemble_params, self.ensemble_buffers = stack_module_state(models)
        self.base_model = copy.deepcopy(models[0])
        self.base_model = self.base_model.to('meta')
    
    def fmodel(self, ensemble_params, ensemble_buffers, x):
        return functional_call(self.base_model, 
                               (ensemble_params, ensemble_buffers), 
                               (x,))
        
    def forward(self,theta):
        pred = vmap(self.fmodel,
                    in_dims=(0,0, None))(self.ensemble_params,
                                      self.ensemble_buffers, 
                                      theta)
        data = torch.mean(pred,axis=0)
        return data
    
class RTFAST(nn.Module):
    """
    This can be called to utilise the ensemble emulator automatically and 
    output only spectra. This will automatically load in scalers and PCA 
    objects required for computation. Instrumental effects are not included.
    
    Input a set of parameters and retrieve the spectrum.
    """
    def __init__(self,device=torch.device('cpu'),num_models=10):
        super().__init__()
        models = [RtdistSpec().to(device) for _ in range(num_models)]
        for i,model in enumerate(models):
            model.load_state_dict(torch.load(f"models/ensemble_{i}.pth",
                                                 map_location=device))
            model.double()
        
        self.ensemble_params, self.ensemble_buffers = stack_module_state(models)
        self.base_model = copy.deepcopy(models[0])
        self.base_model = self.base_model.to('meta')
        
        self.pca  = load("scalers/PCA_20_spec.bin")
        self.comp = load("scalers/comp_20_spec.bin")
        self.spec = load("scalers/spec_20_spec.bin")
        
        self.pca_mean       = torch.Tensor(self.pca.mean_).double()
        self.pca_components = torch.Tensor(self.pca.components_).double()
        self.comp_mean      = torch.Tensor(self.comp.mean_).double()
        self.comp_scale     = torch.Tensor(self.comp.scale_).double()
        self.spec_mean      = torch.Tensor(self.spec.mean_).double()
        self.spec_scale     = torch.Tensor(self.spec.scale_).double()
        
        self.powers         = [0,2,3,4,7,8,10,11,12,13,19]
    
    def fmodel(self, ensemble_params, ensemble_buffers, x):
        return functional_call(self.base_model, 
                               (ensemble_params, ensemble_buffers), 
                               (x,))
    
    def PCA_inverse_transform(self,data_reduced):
        pca_comps = torch.matmul(data_reduced, self.pca_components) + self.pca_mean
        return pca_comps
    
    def comp_inverse_transform(self,data_reduced):
        components = torch.mul(data_reduced,self.comp_scale) + self.comp_mean
        return components
    
    def spec_inverse_transform(self,data_reduced):
        spectra = torch.mul(data_reduced,self.spec_scale) + self.spec_mean
        return spectra
    
    def pars_shift(self,theta):
        #adapts parameters to correct shape of 
        if theta.shape[0] == 20:
            theta[self.powers] = torch.log10(theta[self.powers])
        else:
            theta[:,self.powers] = torch.log10(theta[:,self.powers])
        return theta
    
    def forward(self,theta):
        theta = self.pars_shift(theta)
        pred = vmap(self.fmodel,
                    in_dims=(0,0, None))(self.ensemble_params,
                                      self.ensemble_buffers, 
                                      theta)
        data = torch.mean(pred,axis=0)
        PCA_comps = self.comp_inverse_transform(data)
        std_spec = self.PCA_inverse_transform(PCA_comps)
        spectrum = 10**self.spec_inverse_transform(std_spec)
        return spectrum
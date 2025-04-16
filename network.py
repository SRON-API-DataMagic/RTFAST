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
from torch.distributions import Normal
import os
emudir = os.path.dirname(__file__)

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the
        # latent space
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class DynamicDecoder(nn.Module):
    """
    The decoder portion of the auto-encoder and the front end of the latent
    space to spectrum portion of the emulator.
    """
    
    def __init__(self,spectrum_len,latent_space,nodes,num_layers):
        super().__init__()
        act_type = nn.ReLU()
        modules = []
        modules.append(nn.Linear(latent_space, nodes))
        modules.append(act_type)
        #dynamically add layers
        for i in range(num_layers):
            modules.append(nn.Linear(nodes, nodes))
            modules.append(act_type)
        #add output stack
        modules.append(nn.Linear(nodes, spectrum_len))
        self.LinearStack = nn.Sequential(*modules)
        
    def forward(self,latent_vars):
        return self.LinearStack(latent_vars)

class DynamicEncoder(nn.Module):
    """
    The encoder portion of the auto-encoder.
    """
    
    def __init__(self,spectrum_len,latent_space,nodes,num_layers):
        super().__init__()
        act_type = nn.ReLU()
        modules = []
        modules.append(nn.Linear(spectrum_len, nodes))
        modules.append(act_type)
        #dynamically add layers
        for i in range(num_layers):
            modules.append(nn.Linear(nodes, nodes))
            modules.append(act_type)
        #add output stack
        self.LinearStack = nn.Sequential(*modules)
        self.fc_mean = nn.Linear(nodes, latent_space)
        self.fc_log_var = nn.Linear(nodes, latent_space)
        self.sampling = Sampling()
        
    def forward(self,spectrum):
        out = self.LinearStack(spectrum)
        z_mean = self.fc_mean(out)
        z_log_var = self.fc_log_var(out)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

class DynamicAutoEncoder(nn.Module):
    """
    Full auto-encoder
    """
    def __init__(self,decoder,encoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self,spectrum):
        z_mean, z_log_var, z = self.encoder(spectrum)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction
    
    def latent_space(self,spectrum):
        return self.encoder(spectrum)

class DynamicEmulator(nn.Module):
    """
    Full emulator
    """
    def __init__(self,decoder,encoder):
        super().__init__()
        self.decoder = decoder
        self.par_encoder = encoder
        self.emulator = nn.Sequential(*[self.par_encoder,self.decoder])
    
    def forward(self,pars):
        z = self.par_encoder(pars)
        recon = self.decoder(z)
        return z,recon
    
    def latent_space(self,pars):
        return self.par_encoder(pars)

class RtdistSpec(nn.Module):
    """
    Final neural network emulator architecture. Translates parameters into
    rtdist's time averaged spectrum output. Distinct from the cross-spectrum
    emulator.
    
    Composed of 8 hidden layers, each with 256 nodes. Must be paired with the
    standard scalers and PCA trained with the network to output rtdist
    values directly.
    """
    
    def __init__(self,pars=17,comps=200):
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
        
class RtdistSpec_ensemble(nn.Module):
    def __init__(self,device=torch.device('cpu'),num_models=10):
        super().__init__()
        models = [DynamicNetwork(17,200,12,256).to(device) for _ in range(num_models)]
        for i,model in enumerate(models):
            model.load_state_dict(torch.load(emudir+f"/models/{i}_ensemble.pth",
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
    def __init__(self,device=torch.device('cpu'),num_models=7):
        super().__init__()
        models = [RtdistSpec().to(device) for _ in range(num_models)]
        for i,model in enumerate(models):
            model.load_state_dict(torch.load(emudir+f"/models/{i}_ensemble.pth",
                                                 map_location=device))
            model.double()
        
        self.ensemble_params, self.ensemble_buffers = stack_module_state(models)
        self.base_model = copy.deepcopy(models[0])
        self.base_model = self.base_model.to('meta')
        
        self.pca  = load(emudir+"/scalers/PCA_spec.bin")
        self.comp = load(emudir+"/scalers/comp_spec.bin")
        self.spec = load(emudir+"/scalers/spec_spec.bin")
        
        self.pca_mean       = torch.Tensor(self.pca.mean_).double()
        self.pca_components = torch.Tensor(self.pca.components_).double()
        self.comp_mean      = torch.Tensor(self.comp.mean_).double()
        self.comp_scale     = torch.Tensor(self.comp.scale_).double()
        self.spec_mean      = torch.Tensor(self.spec.mean_).double()
        self.spec_scale     = torch.Tensor(self.spec.scale_).double()
        
        self.powers         = [0,2,3,4,7,8,10,11,12,16]
    
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
        if theta.ndim==1:
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
        spectrum = torch.cat([torch.zeros(50),spectrum])
        return spectrum
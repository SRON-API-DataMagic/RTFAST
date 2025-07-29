"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
import torch
from torch import nn, vmap
from torch.func import stack_module_state, functional_call
from torch.distributions import Normal

import os
import copy
from joblib import load
from scipy.interpolate import interp1d
import numpy as np

import f2py_interface as ib
emudir = os.path.dirname(__file__)

from ndspec.xspec_library import XspecLibrary

def nthcomp(ear,params):
    pass

def tbabs(ear,params):
    pass

lib = XspecLibrary()
lib.initialize_heasoft()
lib.load_models({"tbabs":tbabs,
                 "nthcomp":nthcomp})

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
        nodes = 512
        self.LinearStack = nn.Sequential(nn.Linear(pars, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, nodes),
                                         nn.GELU(),
                                         nn.Linear(nodes, comps))
        
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
            x = x + self.activation(block(x))
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
    def __init__(self,device=torch.device('cpu'),num_models=10):
        super().__init__()
        #load ensemble models
        models = [RtdistSpec(pars=10).to(device) for _ in range(num_models)]
        for i,model in enumerate(models):
            model.load_state_dict(torch.load(emudir+f"/models/rtfast_2_{i}.pth",
                                                 map_location=device))
            model.double()
        self.ensemble_params, self.ensemble_buffers = stack_module_state(models)
        self.base_model = copy.deepcopy(models[0])
        self.base_model = self.base_model.to('meta')
        
        #load scalers
        self.pca  = load(emudir+"/scalers/pca.bin")
        self.comp = load(emudir+"/scalers/pca_scaler.bin")
        self.spec = load(emudir+"/scalers/scaler.bin")
        #transform scaler parameters into pytorch parameters
        self.pca_mean       = torch.Tensor(self.pca.mean_).double()
        self.pca_components = torch.Tensor(self.pca.components_).double()
        self.comp_mean      = torch.Tensor(self.comp.mean_).double()
        self.comp_scale     = torch.Tensor(self.comp.scale_).double()
        self.spec_min      = torch.Tensor(self.spec.min_).double()
        self.spec_scale     = torch.Tensor(self.spec.scale_).double()
        
        #define NN parameter lists and scaling
        self.pars_list = [0,1,2,3,4,6,7,8,9,10]
        self.negatives = [3]
        self.logged = [0,2,3,4,10]
        
        self.powers         = [0,2,3,4,10]
        
        #prepare nthcomp
        self.nthcomp = lib.nthcomp
        self.nthcomp_reltrans_index = [6,10,5]
        self.nthcomp_index = [0,1,4]
        self.nthcomp_par_base = [2,40,0.05,1,0]
        
        #prepare tbabs
        self.tbabs = lib.tbabs
        
        #prepare interal energy grid for interpolation
        self.define_internal_egrid()
    
    def define_internal_egrid(self):
        """
        Defines internal energy grid which RTFAST was built on. This is defined
        between 0.1 and 100.0. Extrapolating outside of this range is at the
        user's own peril.

        Returns
        -------
        None.

        """
        Emin = 0.1
        Emax = 100.0
        ne = 1000
        egrid = np.zeros(ne, dtype = np.float32)
        for i in range(ne):
            egrid[i] = Emin * (Emax/Emin)**(i/ne)
        self.internal_egrid = egrid[:-1]
        return

    
    def fmodel(self, ensemble_params, ensemble_buffers, x):
        """
        Defines a functional model call of the neural network in ensemble form.

        Parameters
        ----------
        ensemble_params : TYPE
            DESCRIPTION.
        ensemble_buffers : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return functional_call(self.base_model, 
                               (ensemble_params, ensemble_buffers), 
                               (x,))
    
    def __PCA_inverse_transform(self,data_reduced):
        pca_comps = torch.matmul(data_reduced, self.pca_components) + self.pca_mean
        return pca_comps
    
    def __comp_inverse_transform(self,data_reduced):
        components = torch.mul(data_reduced,self.comp_scale) + self.comp_mean
        return components
    
    def __spec_inverse_transform(self,data_reduced):
        spectra = torch.mul(data_reduced,self.spec_scale) + self.spec_min
        return spectra
    
    def __pars_shift(self,pars):
        """
        Converts rtdist parameters into neural network friendly form.

        Parameters
        ----------
        negatives: list
            list of indexes of parameters to be turned positive due to being
            a negative value in rtdist
        logged: list
            list of indexes of parameters for their logarithm to be inputted
            into the network

        """
        pars = pars[self.pars_list]
        for i, parameter in enumerate(self.pars_list):
            if parameter in self.negatives:
                if len(pars.shape) > 1:
                    pars[:,i] = -pars[:,i]
                else:
                    pars[i] = -pars[i]
            if parameter in self.logged:
                if len(pars.shape) > 1:
                    pars[:,i] = np.log10(pars[:,i])
                else:
                    pars[i] = np.log10(pars[i])
        return torch.Tensor(pars).double()
    
    def __spectrum_inverse_transform(self,data):
        #Transform PCA components to non-mean scaled form
        PCA_comps = self.__comp_inverse_transform(data)
        #Inverse transform PCA components to log10(spectrum) style data
        std_spec = self.__PCA_inverse_transform(PCA_comps)
        #transform to linear space
        spectrum = 10**self.__spec_inverse_transform(std_spec)
        return spectrum
    
    def reflection_spectrum_prediction(self,egrid,pars):
        #predicted PCA components from NN ensemble
        print("Predicting from ensemble")
        pred = vmap(self.fmodel,
                    in_dims=(0,0, None))(self.ensemble_params,
                                      self.ensemble_buffers, 
                                      pars)
        #averaged PCA components from ensemble
        print("Averaging prediction")
        data = torch.mean(pred,axis=0)
        #inverse transformed to spectrum
        print("Inverse transforming")
        spectrum = self.__spectrum_inverse_transform(data).detach().numpy()
        #interpolate predicted result to inputted energy grid
        print("Interpolating")
        f = interp1d(self.internal_egrid,spectrum,fill_value="extrapolate")
        spectrum = f(egrid)
        return spectrum
    
    def comptonized_continuum(self,egrid,pars,logxi,logne):
        comp = self.nthcomp(egrid,pars)
        Icomp = interp1d(egrid, comp)
        #calculate incident flux in units  [keV/cm^2/s]
        inc_flux = 10**(logne + logxi)/(4.0 * np.pi)/1.602197e-9 
        #renormalise to correct local continuum
        get_norm_cont_local = inc_flux/Icomp/ 1e20
        #return renormalised compton spectrum
        comp = comp * get_norm_cont_local / (10**(logxi + logne - 15))
        return comp
        
    def forward(self,egrid,theta):
        #split input parameters into appropriate parameters for each component
        NN_pars = self.__pars_shift(theta)
        tbabs_pars = theta[11]
        nthcomp_pars = np.copy(self.nthcomp_par_base)
        nthcomp_pars[self.nthcomp_index] = theta[self.nthcomp_reltrans_index]
        boost_par = theta[12]
        logxi = theta[7]
        logne = theta[9]
        
        #predict reflection spectrum from ensemble NN
        print("Beginning prediction")
        spectrum = boost_par*self.reflection_spectrum_prediction(egrid, NN_pars)
        print("Predicted reflection spectrum")
        #add nthcomp component
        spectrum += self.comptonized_continuum(egrid,nthcomp_pars,logxi,logne)
        print("Predicted continuum spectrum")
        #convolve with tbabs absorption
        spectrum *= self.tbabs(egrid,tbabs_pars)
        print("Predicted absorbed spectrum")
        return spectrum

class RTFAST_ensemble(nn.Module):
    """
    This can be called to utilise the ensemble emulator automatically and 
    output only spectra. This will automatically load in scalers and PCA 
    objects required for computation. Instrumental effects are not included.
    
    Input a set of parameters and retrieve the spectrum.
    """
    def __init__(self,device=torch.device('cpu'),num_models=10):
        super().__init__()
        models = [DynamicResNetwork(10,200,6,512).to(device) for _ in range(num_models)]
        for i,model in enumerate(models):
            model.load_state_dict(torch.load(emudir+f"/models/rtfast_2_{i}.pth",
                                                 map_location=device))
            model.double()
        
        self.ensemble_params, self.ensemble_buffers = stack_module_state(models)
        self.base_model = copy.deepcopy(models[0])
        self.base_model = self.base_model.to('meta')
        
        self.pca = load("scalers/pca.bin")
        self.spec = load("scalers/scaler.bin")
        self.comp = load("scalers/pca_scaler.bin")
        
        self.pca_mean       = torch.Tensor(self.pca.mean_).double()
        self.pca_components = torch.Tensor(self.pca.components_).double()
        self.comp_mean      = torch.Tensor(self.comp.mean_).double()
        self.comp_scale     = torch.Tensor(self.comp.scale_).double()
        self.spec_mean      = torch.Tensor(self.spec.mean_).double()
        self.spec_scale     = torch.Tensor(self.spec.scale_).double()
        
        self.powers         = [0,2,3,4,-1]
    
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
        print(data)
        PCA_comps = self.comp_inverse_transform(data)
        std_spec = self.PCA_inverse_transform(PCA_comps)
        spectrum = 10**self.spec_inverse_transform(std_spec)
        return spectrum
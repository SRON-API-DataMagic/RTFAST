"""
This program comprimises the neural network structure used as a base for the
rtdist emulator.
"""
import torch
from torch import nn

from joblib import load
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from importlib.resources import files

emudir = files("rtfast.models")
scalerdir = files("rtfast.scalers")
fortrandir = files("rtfast.fortran")

import ctypes as ct
import sys

type_double_p = ct.POINTER(ct.c_double)

if sys.platform == "darwin":
    lensing = ct.cdll.LoadLibrary(str(fortrandir.joinpath("lensing.so")))
elif sys.platform == "linux" or sys.platform == "linux2":
    lensing = ct.cdll.LoadLibrary(str(fortrandir.joinpath("lensing.dylib")))
else:
    raise RuntimeError("Unsupported platform for Fortran library loading.") 

get_lens = lensing.getlens_
get_lens.argtypes = [type_double_p, type_double_p, type_double_p, type_double_p]
get_lens.restype = None

from ndspec.xspec_library import XspecLibrary 

def nthcomp(ear,params):
    pass

def tbabs(ear,params):
    pass

lib = XspecLibrary()
lib.initialize_heasoft()
lib.load_models({"tbabs":tbabs,
                 "nthcomp":nthcomp})



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
        self.models = [RtdistSpec(pars=10).to(device) for _ in range(num_models)]
        for i,model in enumerate(self.models):
            model.load_state_dict(torch.load(str(emudir.joinpath(f"rtfast_2_{i}.pth")),
                                                 map_location=device))
            model.double()
        
        #retrieves scalers and PCA file info locations
        self.pca  = load(str(scalerdir.joinpath("pca.txt")))
        self.comp = load(str(scalerdir.joinpath("pca_scaler.txt")))
        self.spec = load(str(scalerdir.joinpath("scaler.txt")))
        #reconstructs scalers and PCA
        self.load_scalers()

        #define NN parameter lists and scaling
        self.pars_list = [0,1,2,3,4,6,7,8,9,10]
        self.negatives = [3]
        self.logged = [0,2,3,4,10]
        
        self.powers         = [0,2,3,4,10]
        
        #prepare nthcomp
        self.nthcomp = lib.nthcomp
        self.nthcomp_reltrans_index = [6,10,5]
        self.nthcomp_index = [0,1,4]
        self.nthcomp_par_base = [2,40,0.05,1,0,1]
        
        #prepare tbabs
        self.tbabs = lib.tbabs
        
        #prepare interal energy grid for interpolation
        self.define_internal_egrid()
        self.define_normalisation_grid()
    
    def load_scalers(self):
        """
        Loads in the scalers used to scale the input parameters and PCA
        components. This is required for the neural network to work correctly.
        """
        params = {}
        with open(self.spec) as f:
            for line in f:
                key, *values = line.strip().split()
                if key == "n_features_in":
                    params[key] = int(values[0])
                else:
                    params[key] = np.array([float(v) for v in values])

        self.spec = StandardScaler()
        self.spec.mean_ = params["mean"]
        self.spec.scale_ = params["scale"]
        self.spec.var_ = params["var"]
        self.spec.n_features_in_ = params["n_features_in"]

        params = {}
        with open(self.comp) as f:
            for line in f:
                key, *values = line.strip().split()
                if key == "n_features_in":
                    params[key] = int(values[0])
                else:
                    params[key] = np.array([float(v) for v in values])

        self.comp = MinMaxScaler()
        self.comp.min_ = params["min"]
        self.comp.scale_ = params["scale"]
        self.comp.data_min_ = params["data_min"]
        self.comp.data_max_ = params["data_max"]
        self.comp.data_range_ = params["data_range"]
        self.comp.n_features_in_ = params["n_features_in"]

        params = {}
        with open(self.pca) as f:
            lines = f.readlines()

        def read_matrix(start_idx, n_rows):
            return np.array([list(map(float, lines[i].split())) for i in range(start_idx, start_idx + n_rows)])

        i = 0
        while i < len(lines):
            key = lines[i].strip()
            if key == "components":
                params["components"] = read_matrix(i+1, 200)  # 200 = n_components
                i += 201
            elif key == "mean":
                params["mean"] = np.array(list(map(float, lines[i+1].split())))
                i += 2
            elif key == "explained_variance":
                params["explained_variance"] = np.array(list(map(float, lines[i+1].split())))
                i += 2
            elif key == "n_features_in":
                params["n_features_in"] = int(lines[i+1])
                i += 2
            else:
                i += 1

        self.pca = PCA(n_components=params["components"].shape[0])
        self.pca.components_ = params["components"]
        self.pca.mean_ = params["mean"]
        self.pca.explained_variance_ = params["explained_variance"]
        self.pca.n_features_in_ = params["n_features_in"]

    def define_internal_egrid(self):
        """
        Defines internal energy grid which RTFAST was built on. This is defined
        between 0.1 and 100.0. Extrapolating outside of this range is at the
        user's own peril.
        """
        Emin = 0.1
        Emax = 100.0
        ne = 1000
        egrid = np.zeros(ne, dtype = np.float32)
        for i in range(ne):
            egrid[i] = Emin * (Emax/Emin)**(i/ne)
        self.internal_egrid = egrid[:-1]
        return

    def define_normalisation_grid(self):
        """
        Defines energy grid for normalisation of the continuum (illuminating
        comptonised spectrum nthcomp)

        Returns
        -------
        None.

        """
        nex = 2**12
        Emin = 1e-2
        Emax = 3e3
        self.norm_egrid = np.zeros(nex,dtype=np.float32)
        for i in range(nex):
           self.norm_egrid[i] = Emin * (Emax/Emin)**(i/nex)
        return
    
    def dgsofac(self,a,h):
        """
        Calculates the blue shift experienced by a photon travelling from an 
        on-axis point source to a distant, stationary observer (works for both
        prograde and retrograde spins).

        Parameters
        ----------
        a : float
            spin of the black hole.
        h : float
            height (in Rg) of the source over the black hole.

        Returns
        -------
        None.

        """
        Dh      = h**2 - 2*h + a**2
        dgsofac = Dh / ( h**2 + a**2 )
        dgsofac = np.sqrt( dgsofac )
        return dgsofac
    
    def lensing_factor(self,a,h,muobs):
        a = ct.c_double(a)
        h = ct.c_double(h)
        muobs = ct.c_double(muobs)
        lens = ct.c_double(1)
        # Create ctypes pointers
        a_p = ct.byref(a)
        h_p = ct.byref(h)
        muobs_p = ct.byref(muobs)
        lens_p = ct.byref(lens)
        get_lens(a_p, h_p, muobs_p, lens_p)
        return lens_p._obj.value
    
    def __PCA_inverse_transform(self,data_reduced):
        pca_comps = self.pca.inverse_transform(data_reduced)
        return pca_comps
    
    def __comp_inverse_transform(self,data_reduced):
        components = self.comp.inverse_transform(data_reduced)
        return components
    
    def __spec_inverse_transform(self,data_reduced):
        spectra = self.spec.inverse_transform(data_reduced)
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
                if pars.ndim > 1:
                    pars[:,i] = -pars[:,i]
                else:
                    pars[i] = -pars[i]
            if parameter in self.logged:
                if pars.ndim > 1:
                    pars[:,i] = np.log10(pars[:,i])
                else:
                    pars[i] = np.log10(pars[i])
        return torch.Tensor(pars).double()
    
    def __spectrum_inverse_transform(self,data):
        data = data.detach().numpy()
        if data.ndim == 1:
            data = data.reshape(1, -1)
        #Transform PCA components to non-mean scaled form
        PCA_comps = self.__comp_inverse_transform(data)
        #Inverse transform PCA components to log10(spectrum) style data
        std_spec = self.__PCA_inverse_transform(PCA_comps)
        #transform to linear space
        spectrum = 10**self.__spec_inverse_transform(std_spec)
        return spectrum[0]
    
    def reflection_spectrum_prediction(self,egrid,pars):
        #predicted PCA components from NN ensemble
        pred = torch.stack([model(pars) for model in self.models],dim=0)
        #averaged PCA components from ensemble
        self.pred = pred
        data = torch.mean(pred,axis=0)
        #inverse transformed to spectrum
        spectrum = self.__spectrum_inverse_transform(data)
        #interpolate predicted result to inputted energy grid
        f = interp1d(self.internal_egrid,spectrum,fill_value="extrapolate")
        spectrum = f(egrid)
        return spectrum
    
    def calculate_normalisation(self,pars):
        earx = self.norm_egrid
        norm_comp = self.nthcomp(earx,pars)
        Icomp = 0
        for i in range(1,len(norm_comp)):
           E   = 0.5 * ( earx[i] + earx[i-1] )
           if (E >= 0.1 and E <= 1e3):
              Icomp = Icomp + ((earx[i] + earx[i-1]) * 0.5 * norm_comp[i])
        return Icomp
    
    def comptonized_continuum(self,egrid,pars,logxi,logne):
        egrid,pars = egrid.astype(np.float32),pars.astype(np.float32)
        comp = self.nthcomp(egrid,pars)
        Icomp = self.calculate_normalisation(pars)
        #calculate incident flux in units  [keV/cm^2/s]
        inc_flux = (10**(logne + logxi)) /(4.0 * np.pi* 1.602197e-9)
        #renormalise to correct local continuum
        get_norm_cont_local = inc_flux/Icomp/ 1e20
        #return renormalised compton spectrum
        comp = comp * get_norm_cont_local / (10**(logxi + logne - 15))
        return comp
        
    def forward(self,egrid,theta):
        #split input parameters into appropriate parameters for each component
        dgsofac = self.dgsofac(theta[1],theta[0])
        inc = theta[2]
        muobs = np.cos(inc*np.pi/180) 
        boost = theta[12]
        logxi = theta[7]
        logne = theta[9]
        z = theta[5]
        kTe = theta[10]
        lens = self.lensing_factor(theta[1], theta[0], muobs)
        theta[10] = (theta[10]*dgsofac)/(1+z) #redefines temperature to be observed electron temperature for NN
        NN_pars = self.__pars_shift(theta)
        tbabs_pars = theta[11]
        nthcomp_pars = np.copy(self.nthcomp_par_base)
        nthcomp_pars[self.nthcomp_index] = theta[self.nthcomp_reltrans_index]
        nthcomp_pars[1] = kTe
        nthcomp_pars[4] = (1.0/ dgsofac) - 1.0
        #predict reflection spectrum from ensemble NN
        spectrum = (np.abs(boost)*self.reflection_spectrum_prediction(egrid, NN_pars))[:-1]
        #add nthcomp component
        if boost >= 0:
            comp = self.comptonized_continuum(egrid,nthcomp_pars,logxi,logne)
            comp = lens * (dgsofac/(1+z)) * comp
            spectrum += comp
        #convolve with tbabs absorption
        spectrum *= self.tbabs(egrid.astype(np.float32),np.array([tbabs_pars],dtype=np.float32))
        return spectrum
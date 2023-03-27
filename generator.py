"""
This program deals with generating rtdist models with given parameters. For use
for generating sampels for training the emulator for rtdist.
"""
import numpy as np
from reltrans import _models

def rtdist_flux(pars,egrid):
    """
    

    Parameters
    ----------
    pars : array
        contains parameters used in simulation.
    egrid : array
        contains values of energy to calculate for.

    Returns
    -------
    model : array
        outputted simulated data.

    """
    model = _models.tdrtdist(pars,egrid)
    return model

def rtdist_lags(pars,egrid):
    """
    

    Parameters
    ----------
    pars : array
        contains parameters used in simulation.
    egrid : array
        contains values of energy to calculate for. Final point in array will
        always be zero when evaluated due to quirk in Sherpa/Xspec.

    Returns
    -------
    output : array
        outputted simulated data.

    """
    y = _models.tdrtdist(pars,egrid)
    dE = np.diff(egrid)
    output = y[:-1]/dE
    return output


def pregen():
    """
    Generates random parameters for the rtdist model to evaluate

    Returns
    -------
    pars : list
        list of parameters for evaluation by rtdist.

    """
    pars = [6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,0,0,0.95,0,
            -0.8,0.3,2.2e-4,1,1.]
    uni = np.random.uniform
    a = uni(0.1,0.998)
    mass = 10**uni(1,11)
    
    pars[1] = a
    pars[13] = mass
    
    return pars

def lhs_range_gen():
    height_range = [np.log10(1.3),np.log10(1e4)]
    spin_range = [0.1,0.998]
    inclination_range = [1,80]
    r_inner_range = [-400,-1]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    z_range = [0,4]
    Gamma_range = [1.4,3.4]
    distance_range = [np.log10(0.2),np.log10(1e10)]
    Afe_range = [0.5,10]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    nH_range = [np.log10(1e-22),np.log10(1e6)]
    boost_range = [np.log10(1e-2),np.log10(10)]
    mass_range = [np.log10(1),np.log10(1e11)]
    honr_range = [0,0.176]
    b1_range = [0,2]
    b2_range = [-4,4]
    phiAB_range = [-3.14,3.14]
    g_range = [0,0.5]
    Anorm_range = [np.log10(1e-12),np.log10(1e10)]
    
    
    range_all = [height_range,spin_range,inclination_range,r_inner_range,
                 r_outer_range,z_range,Gamma_range,distance_range,Afe_range,
                 logNe_range,kte_range,nH_range,boost_range,mass_range,
                 honr_range,b1_range,b2_range,phiAB_range,g_range,Anorm_range]
    
    return range_all

def pars_conversion(pars):
    """
    Converts sampled parameters for neural network training into correct
    format for use in generating data and adds non-sampled parameters
    needed by the model

    Parameters
    ----------
    pars : np.ndarray
        large array that contains sampled parameters.

    Returns
    -------
    pars : np.ndarray
        large array that contains correctly formatted parameters ready for 
        parsing into external model.

    """
    #convert all log spaced parameters into their linear form
    pars[:,0] = 10**pars[:,0]
    pars[:,4] = 10**pars[:,4]
    pars[:,7] = 10**pars[:,7]
    pars[:,9:13] = 10**pars[:,9:13]
    pars[:,18] = 10**pars[:,18]
    #add fixed parameters
    pars = np.insert(pars,17,0,axis=1) #fmin
    pars = np.insert(pars,18,0,axis=1) #fmax
    pars = np.insert(pars,19,1,axis=1) #ReIm
    pars = np.insert(pars,20,0,axis=1) #phiA
    pars = np.insert(pars,24,1,axis=1) #RESP
    pars = np.insert(pars,25,0,axis=1) #xspec norm
    
    return pars
    
    
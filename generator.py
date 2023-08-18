"""
This program deals with generating rtdist models with given parameters. For use
for generating sampels for training the emulator for rtdist.
"""
import numpy as np
from reltrans import _models
from joblib import Parallel, delayed
from processing import nanChecker, saveData
from sklearn.preprocessing import MinMaxScaler
import scipy
from dataStructures import FluxData, LagsData

def rtdist_flux(pars, egrid):
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
    model = _models.tdrtdist(pars, egrid)
    return model

def rtdist_lags(pars, egrid):
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
    y = _models.tdrtdist(pars, egrid)
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
    """
    Generates valid ranges of parameters to be trained on

    Returns
    -------
    range_all : list
        gives parameter ranges for each of the given parameters listed. Used 
        in the latin hypercube sampling

    """
    height_range = [np.log10(4),np.log10(1e4)]
    spin_range = [0.1,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    z_range = [0,4]
    Gamma_range = [1.4,3.4]
    distance_range = [np.log10(0.2),np.log10(1e10)]
    Afe_range = [np.log10(0.5),np.log10(10)]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    nH_range = [np.log10(1e-22),np.log10(1e6)]
    boost_range = [np.log10(1e-2),np.log10(10)]
    mass_range = [np.log10(3),np.log10(1e11)]
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

def lhs_trimmed_gen():
    """
    Limited form of lhs_range_gen that returns ranges for only a limited amount
    of parameters. Used in the comparitive between grid and active learning
    strategies.

    Returns
    -------
    range_all : list
        a list of ranges of parameter spaces to generate from.

    """
    spin_range = [0.1,0.998]
    mass_range = [np.log10(3),np.log10(1e11)]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    
    range_all = [spin_range,mass_range,inclination_range,r_inner_range,r_outer_range]
    
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
    pars_base = [6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,0,
                 0,0,0.95,-0.8,0.3,2.2e-4,1,1.]
    new_pars = []
    for i in range(pars.shape[0]):
        new_pars.append(pars_base)
    new_pars = np.asarray(new_pars)
    new_pars[:,1] = pars[:,0]
    new_pars[:,13] = 10**pars[:,1]
    new_pars[:,2] = 10**pars[:,2]
    new_pars[:,3] = -10**pars[:,3]
    new_pars[:,4] = 10**pars[:,4]
    
    return new_pars

def pars_conversion_full(pars):
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
    pars_base = [6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,0,
                 0,0,0.95,-0.8,0.3,2.2e-4,1,1.]
    new_pars = []
    for i in range(pars.shape[0]):
        new_pars.append(pars_base)
    new_pars = np.asarray(new_pars)
    new_pars[:,0] = -10**pars[:,0]
    new_pars[:,1] = pars[:,1]
    new_pars[:,2] = 10**pars[:,2]
    new_pars[:,3] = -10**pars[:,3]
    new_pars[:,4] = 10**pars[:,4]
    new_pars[:,5] = pars[:,5]
    new_pars[:,6] = pars[:,6]
    new_pars[:,7] = 10**pars[:,7]
    new_pars[:,8] = 10**pars[:,8]
    new_pars[:,9] = pars[:,9]
    new_pars[:,10] = 10**pars[:,10]
    new_pars[:,11] = 10**pars[:,11]
    new_pars[:,12] = 10**pars[:,12]
    new_pars[:,13] = 10**pars[:,13]
    new_pars[:,14] = pars[:,14]
    new_pars[:,15] = pars[:,15]
    new_pars[:,16] = pars[:,16]
    new_pars[:,21] = pars[:,17]
    new_pars[:,22] = pars[:,18]
    new_pars[:,23] = 10**pars[:,19]
    
    return new_pars

def grid_data_gen(size, fname, egrid, lags_egrid):
    """
    Creates a grid of parameter space and then generates spectra and time lags
    for each spot on the grid. Saves these to the disk

    Parameters
    ----------
    size : int
        how many grid points in the parameter space to generate.
    fname : string
        signifies the file name to save as.
    egrid : ndarray or list
        energy grid to pass into rtdist for the spectra to be generated on.
    lags_egrid : ndarray or list
        energy grid to pass into rtdist for the time lags to be generated on.

    Returns
    -------
    None.

    """
   
    spin = np.linspace(0.1,1.0,size)
    mass = np.linspace(np.log10(3.3),np.log10(1e11),size)
    inc = np.linspace(np.log10(1),np.log10(80),size)
    r_in = np.linspace(np.log10(1),np.log10(400),size)
    r_out = np.linspace(np.log10(400),np.log10(1e5),size)
    
    #create parameter grid
    theta_init = []
    for a in spin:
        for m in mass:
            for i in inc:
                for r_i in r_in:
                    for r_o in r_out:
                        theta_init.append([a,m,i,r_i,r_o])
    theta_init = np.asarray(theta_init)
    #convert to rtdist model compatible parameters
    pars_init = pars_conversion(theta_init)
    with Parallel(n_jobs=10,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        flux_data_init = parallel(delayed(rtdist_flux)(pars, egrid)
                                        for pars in pars_init)
        lags_data_init = parallel(delayed(rtdist_lags)(pars, lags_egrid)
                                        for pars in pars_init)
    flux_data_init = np.array(flux_data_init)
    lags_data_init = np.array(lags_data_init)
    
    indexes = nanChecker(flux_data_init, pars_init)
    flux_data_init = np.delete(flux_data_init,indexes, axis=0)
    lags_data_init = np.delete(lags_data_init,indexes, axis=0)
    pars_init = np.delete(pars_init,indexes, axis=0)
    
    indexes = nanChecker(lags_data_init, pars_init)
    flux_data_init = np.delete(flux_data_init,indexes, axis=0)
    lags_data_init = np.delete(lags_data_init,indexes, axis=0)
    pars_init = np.delete(pars_init,indexes, axis=0)
    
    idxs = np.arange(0,flux_data_init.shape[0])
    np.random.shuffle(idxs)
    tra_idx = idxs[:int(0.9*len(idxs))]
    tes_idx = idxs[int(0.9*len(idxs)):]
    
    #Splitting data and parameters into training and testing datasets
    train_data = flux_data_init[tra_idx]
    train_lags = lags_data_init[tra_idx]
    train_pars = pars_init[tra_idx]
    test_data = flux_data_init[tes_idx]
    test_lags = lags_data_init[tes_idx]
    test_pars = pars_init[tes_idx]
    
    print("Saving to disk")
    #save data for the first time in text files
    saveData(train_data, train_pars, 
             "data/locations/",f"loc_{fname}_flux.csv")
    saveData(train_lags, train_pars, 
             "data/locations/",f"loc_{fname}_lags.csv")
    saveData(test_data, test_pars, 
             "data/locations/",f"loc_{fname}_flux_test.csv")
    saveData(test_lags, test_pars, 
             "data/locations/",f"loc_{fname}_lags_test.csv")
    
    scaler = MinMaxScaler()
    
    flux_dataloader = FluxData(f"data/locations/loc_{fname}_flux.csv", 
                                   scaler,f"{fname}_flux_scaler.bin",
                                   scaling=True)
    lags_dataloader = LagsData(f"data/locations/loc_{fname}_lags.csv", 
                                   scaler,f"{fname}_lags_scaler.bin",
                                   scaling=True)
    
    return   

def generate_test_set(size, egrid, lags_egrid):
    """
    

    Parameters
    ----------
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    range_all = np.asarray(lhs_trimmed_gen())
    #pre generate Latin Hypercube samples.
    sampler = scipy.stats.qmc.LatinHypercube(d=len(range_all))
    sample = sampler.random(n=size)
    theta_lhs = scipy.stats.qmc.scale(sample, range_all[:,0], range_all[:,1])

    #generate physical models of test set
    theta_lhs_iterate = pars_conversion(theta_lhs)
    with Parallel(n_jobs=10,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        data_init = parallel(delayed(rtdist_flux)(pars, egrid)
                                        for pars in theta_lhs_iterate)
        lags = parallel(delayed(rtdist_lags)(pars, lags_egrid)
                                        for pars in theta_lhs_iterate)
    data_init = np.asarray(data_init)
    lags = np.asarray(lags)
    return data_init, lags, theta_lhs_iterate
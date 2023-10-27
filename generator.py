"""
This program deals with generating rtdist models with given parameters. For use
for generating sampels for training the emulator for rtdist.
"""
import numpy as np
import os
from reltrans import _models
from joblib import Parallel, delayed
from processing import nanChecker, saveData, mergeSaveData, renameData
from processing import readAndRemoveNans, spectraChecker
from sklearn.preprocessing import MinMaxScaler
import scipy
from dataStructures import FluxData, LagsData
import pandas as pd
from sherpa.astro.ui import unpack_rmf

def rtdist_erg_flux(pars, egrid):
    """
    

    Parameters
    ----------
    model : sherpa model
        model that you wish to integrate energy flux over 2-10keV.

    Returns
    -------
    flux : float
        return energy flux between 2-10keV.

    """
    model = _models.tdrtdist(pars, egrid)
    #take middle energy and multiply all photon counts by said energy
    gmid = 1.60217653e-09  * (egrid[:-1] + egrid[1:]) / 2
    model = model[:-1]*gmid
    start = np.argmin(np.abs(egrid-2))
    end = np.argmin(np.abs(egrid-10))
    flux = model[start:end].sum()
    return flux

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

def lhc_filter(lhc):
    """
    Removes unphysical parameter sets from the Latin Hypercube. This prevents
    overly bright sources from being generated as well as reducing time spent
    on generating model data for objects that we won't see

    Parameters
    ----------
    lhc : np.ndarray
        latin hypercube containing parameter sets for rtdist.

    Returns
    -------
    new_lhc : np.ndarray
        latin hypercube with unphysical parameter sets removed..

    """
    #bad sets indexes all parameter sets that don't fit the filter criteria
    #removes parameter sets that have a ph0ton index higher than 3 AND a iron
    #solar abundance above 6 AND a electron density in the disk of higher than
    #10^19.
    bad_sets = np.nonzero((lhc[:,6]>2.75)&(10**lhc[:,8]>4)&(lhc[:,9]>19))
    #removes all unphysical sets from the parameter sets
    new_lhc = np.delete(lhc,bad_sets,0)
    return new_lhc

def lhc_all():
    """
    Generates valid ranges of parameters of AGN to be trained on
    
    Returns
    -------
    range_all : list
        gives parameter ranges for each of the given parameters listed. Used 
        in the latin hypercube sampling
    
    """
    height_range = [np.log10(1.5),np.log10(1e4)]
    spin_range = [0,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    z_range = [0,4]
    Gamma_range = [1.4,3.4]
    distance_range = [np.log10(0.2),np.log10(1e10)]
    Afe_range = [np.log10(0.5),np.log10(10)]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    nH_range = [np.log10(1e-3),np.log10(1e3)]
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

def lhc_BH():
    """
    Generates valid ranges of parameters of stellar mass BHs to be trained on

    Returns
    -------
    range_all : list
        gives parameter ranges for each of the given parameters listed. Used 
        in the latin hypercube sampling

    """
    height_range = [np.log10(1.5),np.log10(1e4)]
    spin_range = [0,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    z_range = [0,4]
    Gamma_range = [1.4,3.4]
    distance_range = [np.log10(0.2),np.log10(3e4)]
    Afe_range = [np.log10(0.5),np.log10(10)]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    nH_range = [np.log10(1e-3),np.log10(1e3)]
    boost_range = [np.log10(1e-2),np.log10(10)]
    mass_range = [np.log10(3),np.log10(40)]
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

def lhc_AGN():
    """
    Generates valid ranges of parameters of AGN to be trained on.
    
    Returns
    -------
    range_all : list
        gives parameter ranges for each of the given parameters listed. Used 
        in the latin hypercube sampling
    
    """
    height_range = [np.log10(1.5),np.log10(1e4)]
    spin_range = [0,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    z_range = [0,4]
    Gamma_range = [1.4,3.4]
    distance_range = [np.log10(3.5e6),np.log10(1e10)]
    Afe_range = [np.log10(0.5),np.log10(10)]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    nH_range = [np.log10(1e-3),np.log10(1e3)]
    boost_range = [np.log10(1e-2),np.log10(10)]
    mass_range = [np.log10(1e4),np.log10(1e11)]
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

def lhc_explor():
    """
    Generates valid ranges of parameters of AGN to be trained on. This is
    a function purely used for exploring parameter ranges and generally not
    intended for use in training of the emulator.
    
    Returns
    -------
    range_all : list
        gives parameter ranges for each of the given parameters listed. Used 
        in the latin hypercube sampling
    
    """
    height_range = [np.log10(1.5),np.log10(100)]
    spin_range = [0,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(100)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    z_range = [0,4]
    Gamma_range = [1.4,3.4]
    distance_range = [np.log10(3.5e6),np.log10(1e10)]
    Afe_range = [np.log10(0.5),np.log10(10)]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    nH_range = [np.log10(1e-3),np.log10(1e3)]
    boost_range = [np.log10(1e-2),np.log10(10)]
    mass_range = [np.log10(1e4),np.log10(1e11)]
    honr_range = [0,0.176]
    b1_range = [0,2]
    b2_range = [-4,4]
    phiAB_range = [-3.14,3.14]
    g_range = [0,0.5]
    Anorm_range = [np.log10(1e-5),np.log10(1e5)]
    
    
    range_all = [height_range,spin_range,inclination_range,r_inner_range,
                 r_outer_range,z_range,Gamma_range,distance_range,Afe_range,
                 logNe_range,kte_range,nH_range,boost_range,mass_range,
                 honr_range,b1_range,b2_range,phiAB_range,g_range,Anorm_range]
    
    return range_all

def lhc_trimmed_gen():
    """
    Limited form of lhc_range_gen that returns ranges for only a limited amount
    of parameters. Used in the comparitive between grid and active learning
    strategies.

    Returns
    -------
    range_all : list
        a list of ranges of parameter spaces to generate from.

    """
    spin_range = [0.1,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    distance_range = [np.log10(0.2),np.log10(1e10)]
    
    range_all = [spin_range,inclination_range,r_inner_range,r_outer_range,
                 distance_range]
    
    return range_all

def pars_conversion(pars,ReIm):
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
                 0,ReIm,0,-0.8,0.3,2.2e-4,1,1.]
    new_pars = []
    for i in range(pars.shape[0]):
        new_pars.append(pars_base)
    new_pars = np.asarray(new_pars)
    new_pars[:,1] = pars[:,0]
    new_pars[:,2] = 10**pars[:,1]
    new_pars[:,3] = -10**pars[:,2]
    new_pars[:,4] = 10**pars[:,3]
    new_pars[:,13] = 10**pars[:,4]
    
    return new_pars

def pars_conversion_full(pars,ReIm):
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
                 0,ReIm,0,-0.8,0.3,2.2e-4,1,1.]
    new_pars = []
    for i in range(pars.shape[0]):
        new_pars.append(pars_base)
    new_pars = np.asarray(new_pars)
    new_pars[:,0] = -10**pars[:,0]  #height
    new_pars[:,1] = pars[:,1]       #spin
    new_pars[:,2] = 10**pars[:,2]   #inclination
    new_pars[:,3] = -10**pars[:,3]  #inner radius
    new_pars[:,4] = 10**pars[:,4]   #outer radius
    new_pars[:,5] = pars[:,5]       #redshift (z)
    new_pars[:,6] = pars[:,6]   #Gamma
    new_pars[:,7] = 10**pars[:,7]   #distance
    new_pars[:,8] = 10**pars[:,8]   #Afe
    new_pars[:,9] = pars[:,9]       #logNe
    new_pars[:,10] = 10**pars[:,10] #kTe
    new_pars[:,11] = 10**pars[:,11] #nH
    new_pars[:,12] = 10**pars[:,12] #boost
    new_pars[:,13] = 10**pars[:,13] #mass
    new_pars[:,14] = pars[:,14]     #scale height of disk
    new_pars[:,15] = pars[:,15]     #b1
    new_pars[:,16] = pars[:,16]     #b2
    new_pars[:,21] = pars[:,17]     #phiAB
    new_pars[:,22] = pars[:,18]     #coherence
    new_pars[:,23] = 10**pars[:,19] #Anorm
    
    return new_pars

def pars_conversion_explor(pars,ReIm):
    """
    Converts sampled parameters for neural network training into correct
    format for use in generating data and adds non-sampled parameters
    needed by the model. This is primarily used for exploring the effects
    of parameter ranges on flux so as to further restrict the parameter space.
    Not intended for use in training of the emulator.

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
                 0,ReIm,0,-0.8,0.3,2.2e-4,1,1.]
    new_pars = []
    for i in range(pars.shape[0]):
        new_pars.append(pars_base)
    new_pars = np.asarray(new_pars)
    new_pars[:,0] = -10**pars[:,0]  #height
    new_pars[:,1] = pars[:,1]       #spin
    new_pars[:,2] = 10**pars[:,2]   #inclination
    new_pars[:,3] = -10**pars[:,3]  #inner radius
    new_pars[:,4] = 10**pars[:,4]   #outer radius
    new_pars[:,5] = pars[:,5]       #redshift (z)
    new_pars[:,6] = pars[:,6]   #Gamma
    new_pars[:,7] = 10**pars[:,7]   #distance
    new_pars[:,8] = 10**pars[:,8]   #Afe
    new_pars[:,9] = pars[:,9]       #logNe
    new_pars[:,10] = 10**pars[:,10] #kTe
    new_pars[:,11] = 10**pars[:,11] #nH
    new_pars[:,12] = 10**pars[:,12] #boost
    new_pars[:,13] = 10**pars[:,13] #mass
    new_pars[:,14] = pars[:,14]     #scale height of disk
    new_pars[:,15] = pars[:,15]     #b1
    new_pars[:,16] = pars[:,16]     #b2
    new_pars[:,21] = pars[:,17]     #phiAB
    new_pars[:,22] = pars[:,18]     #coherence
    new_pars[:,23] = 1              #Anorm
    
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
    inc = np.linspace(np.log10(1),np.log10(80),size)
    r_in = np.linspace(np.log10(1),np.log10(400),size)
    r_out = np.linspace(np.log10(400),np.log10(1e5),size)
    distance = np.linspace(np.log10(400),np.log10(1e5),size)
    
    #create parameter grid
    theta_init = []
    for a in spin:
        for i in inc:
                for r_i in r_in:
                    for r_o in r_out:
                        for d in distance:
                            theta_init.append([a,i,r_i,r_o,d])
    theta_init = np.asarray(theta_init)
    #convert to rtdist model compatible parameters
    theta_flux = pars_conversion(theta_init,0)
    theta_lags = pars_conversion(theta_init,6)
    
    with Parallel(n_jobs=10,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        flux_data_init = parallel(delayed(rtdist_flux)(pars, egrid)
                                        for pars in theta_flux)
        lags_data_init = parallel(delayed(rtdist_lags)(pars, lags_egrid)
                                        for pars in theta_lags)
    flux_data_init = np.array(flux_data_init)
    lags_data_init = np.array(lags_data_init)
    
    indexes = nanChecker(flux_data_init, theta_flux)
    flux_data_init = np.delete(flux_data_init,indexes, axis=0)
    lags_data_init = np.delete(lags_data_init,indexes, axis=0)
    theta_flux = np.delete(theta_flux,indexes, axis=0)
    theta_lags = np.delete(theta_lags,indexes, axis=0)
    
    indexes = nanChecker(lags_data_init, theta_lags)
    flux_data_init = np.delete(flux_data_init,indexes, axis=0)
    lags_data_init = np.delete(lags_data_init,indexes, axis=0)
    theta_flux = np.delete(theta_flux,indexes, axis=0)
    theta_lags = np.delete(theta_lags,indexes, axis=0)
    
    idxs = np.arange(0,flux_data_init.shape[0])
    np.random.shuffle(idxs)
    tra_idx = idxs[:int(0.9*len(idxs))]
    tes_idx = idxs[int(0.9*len(idxs)):]
    
    #Splitting data and parameters into training and testing datasets
    train_data = flux_data_init[tra_idx]
    train_lags = lags_data_init[tra_idx]
    train_flux_pars = theta_flux[tra_idx]
    train_lags_pars = theta_lags[tra_idx]
    
    test_data = flux_data_init[tes_idx]
    test_lags = lags_data_init[tes_idx]
    test_flux_pars = theta_flux[tes_idx]
    test_lags_pars = theta_lags[tes_idx]
    
    print("Saving to disk")
    #save data for the first time in text files
    saveData(train_data, train_flux_pars, 
             "data/locations/",f"loc_{fname}_flux.csv")
    saveData(train_lags, train_lags_pars, 
             "data/locations/",f"loc_{fname}_lags.csv")
    saveData(test_data, test_flux_pars, 
             "data/locations/",f"loc_{fname}_flux_test.csv")
    saveData(test_lags, test_lags_pars, 
             "data/locations/",f"loc_{fname}_lags_test.csv")
    
    scaler = MinMaxScaler()
    
    pars_list = [1,2,3,4,7]
    negatives = [2]
    logged = [1,2,3,4]
    
    flux_dataloader = FluxData(f"data/locations/loc_{fname}_flux.csv", 
                                   scaler,f"{fname}_flux_scaler.bin",
                                   pars_list=pars_list,
                                   negatives=negatives, logged=logged, 
                                   scaling=True)
    lags_dataloader = LagsData(f"data/locations/loc_{fname}_lags.csv", 
                                   scaler,f"{fname}_lags_scaler.bin", 
                                   pars_list=pars_list,
                                   negatives=negatives, logged=logged, 
                                   scaling=True)
    
    return   

def generate_flux_dists(AGN_name):
    wrk_dir = os.getcwd()
    rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
    rmf = unpack_rmf(rmf_name)
    egrid = rmf.e_min #energy grid used to evaluate the xspec model
    
    labels = ["height","a","inc","rin","rout","z","Gamma","Dkpc","Afe",
              "logNe","kte","nH","boost","mass","honr","b1","b2","phiAB","g",
              "Anorm"]
    range_AGN = np.asarray(lhc_explor())
    
    #pre generate Latin Hypercube samples.
    theta_agn = lhc_generation(int(1e3), range_AGN)
    theta_agn = lhc_filter(theta_agn)
    
    iter_agn = pars_conversion_explor(theta_agn, 0)
    
    with Parallel(n_jobs=10,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        AGN_flux = parallel(delayed(rtdist_erg_flux)(pars, egrid)
                                        for pars in iter_agn)
    AGN_flux = np.asarray(AGN_flux)
    
    AGN = pd.DataFrame(data=theta_agn,columns=labels)
    AGN["flux"] = AGN_flux
    
    AGN.to_csv(f"data/flux/{AGN_name}.csv")
    return
    
def generate_test_set(size, egrid, lags_egrid, lhc_gen):
    """
    

    Parameters
    ----------
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    range_all = np.asarray(lhc_gen())
    #pre generate Latin Hypercube samples.
    sampler = scipy.stats.qmc.LatinHypercube(d=len(range_all))
    sample = sampler.random(n=size)
    theta_lhc = scipy.stats.qmc.scale(sample, range_all[:,0], range_all[:,1])

    #generate physical models of test set
    theta_flux = pars_conversion_full(theta_lhc,0)
    theta_lags = pars_conversion_full(theta_lhc,6)
    
    with Parallel(n_jobs=10,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        flux = parallel(delayed(rtdist_flux)(pars, egrid)
                                        for pars in theta_flux)
        flux, theta_flux, theta_lags = spectraChecker(flux,theta_flux,theta_lags,
                                                      1e-11)
        lags = parallel(delayed(rtdist_lags)(pars, lags_egrid)
                                        for pars in theta_lags)
    flux = np.asarray(flux)
    lags = np.asarray(lags)
    return flux, lags, theta_flux, theta_lags

def active_learning_generation(theta_query, egrid, lags_egrid, parallel, 
                               flux_name, flux_test_name, lags_name,
                               lags_test_name,pars_conversion = pars_conversion_full):
    # compute the physical model for these thetas
    theta_flux = pars_conversion_full(theta_query,0)
    theta_lags = pars_conversion_full(theta_query,6)
    print("Generating flux models")
    flux =  parallel(delayed(rtdist_flux)(pars, egrid)
                                    for pars in theta_flux)
    flux = np.asarray(flux)
    print("Checking flux models are valid")
    flux, theta_flux, theta_lags = spectraChecker(flux,theta_flux,theta_lags,
                                                  1e-11)
    print("Saving flux data")
    saveData(flux, theta_flux, 
             "data/locations/","active_gen_flux.csv")
    del flux
    print("Generating lags models")
    lags =  parallel(delayed(rtdist_lags)(pars, lags_egrid)
                                    for pars in theta_lags)
    lags = np.asarray(lags)
    print("Saving lags data")
    saveData(lags, theta_lags, 
             "data/locations/","active_gen_lags.csv", lags=True)
    del theta_flux, theta_lags, lags
    
    print("Performing data cleanup")
    readAndRemoveNans("data/locations/active_gen_flux.csv", 
                      "data/locations/active_gen_lags.csv")
    
    flux = pd.read_csv("data/locations/active_gen_flux.csv")
    lags = pd.read_csv("data/locations/active_gen_lags.csv")
    
    # shuffle indices for neural network training
    idx_shuffle = np.arange(0, len(flux), dtype=int)
    np.random.shuffle(idx_shuffle)

    idx_query = idx_shuffle[:len(idx_shuffle)-250]
    idx_test = idx_shuffle[-250:]
    
    #Split data into test and training sets
    flux_test = flux.iloc[idx_test]
    lags_test = lags.iloc[idx_test]
    
    flux_query = flux.iloc[idx_query]
    lags_query = lags.iloc[idx_query]
    
    #save final curated datasets back to disk for use
    mergeSaveData(flux_query, pd.read_csv(f"data/locations/{flux_name}"),
                  "data/locations/", flux_name)
    mergeSaveData(lags_query, pd.read_csv(f"data/locations/{lags_name}"),
                  "data/locations/", lags_name)
    
    renameData(flux_test, "data/locations/", 
               flux_test_name)
    renameData(lags_test, "data/locations/", 
               lags_test_name)
    return

def lhc_generation(size,range_all):
    sampler = scipy.stats.qmc.LatinHypercube(d=len(range_all))
    sample = sampler.random(n=size)
    theta_lhc = scipy.stats.qmc.scale(sample, range_all[:,0], range_all[:,1])
    return theta_lhc

def intialize_dataset(theta_lhc,egrid,lags_egrid,flux_name,lags_name):
    print("Generating first time dataset")
    init_data_size = 5000
    lhc_idx = init_data_size
    #generating a random set of parameters and corresponding data
    theta_init = theta_lhc[:init_data_size]
    theta_flux = pars_conversion_full(theta_init,0)
    theta_lags = pars_conversion_full(theta_init,6)
    print("Parallelized model generation")
    
    print("Generating flux models")
    flux =  Parallel(n_jobs=10,verbose=5)(delayed(rtdist_flux)(pars, egrid)
                                    for pars in theta_flux)
    flux = np.asarray(flux)
    print("Checking for spectra below threshold")
    flux, theta_flux, theta_lags = spectraChecker(flux,theta_flux,theta_lags,
                                                  1e-11)
    print("Saving flux data")
    saveData(flux, theta_flux, 
             "data/locations/",flux_name)
    
    del flux
    print("Generating lags models")
    lags_query =  Parallel(n_jobs=10,verbose=5)(delayed(rtdist_lags)(pars, lags_egrid)
                                    for pars in theta_lags)
    lags_query = np.asarray(lags_query)
    print("Saving lags data")
    saveData(lags_query, theta_lags,"data/locations/",lags_name, lags=True)
    del theta_flux, theta_lags, lags_query
    
    print("Performing data cleanup")
    readAndRemoveNans(f"data/locations/{flux_name}", 
                      f"data/locations/{lags_name}")
    return lhc_idx
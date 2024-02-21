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

def Anorm_wrapper(pars):
    """
    This function recalculates the Anorm input parameter of rtdist from the 
    generated values of flux. This allows considerably more efficient sampling
    of Anorm - only physically plausible scenarios are considered and we 
    have physical reasoning behind the Anorm values chosen.

    Parameters
    ----------
    pars : np.ndarray
        the curated hypercube of parameters that the emulator will eventually
        be trained on.

    Returns
    -------
    new_lhc : np.ndarray
        latin hypercube with "coronal flux" parameter replaced with Anorm.

    """
    #calculate g_0
    h = 10**pars[:,0]
    a = pars[:,1]
    Dh = h**2 -2*h +a**2
    g_so = np.sqrt(Dh/(h**2 + a**2))
    #calculate luminosity of corona
    F = 10**pars[:,19]              #Flux of corona in erg/cm^2/s
    gamma = pars[:,6]               #photon index
    Afe = pars[:,8]
    z = pars[:,5]
    refl_frac = np.zeros(gamma.shape)
    inc = pars[:,2]
    #calculate normalisation for each flux spectra
    integrals = np.zeros(gamma.shape)
    #cutoff in keV
    E_cut = np.ones(gamma.shape)*300
    logxi = np.ones(gamma.shape)
    xnorm = np.ones(gamma.shape)
    xill_pars = np.vstack((gamma,Afe,E_cut,logxi,z,inc,refl_frac,xnorm)).T
    #energies from 0.1keV to 1MeV
    bins = 1000
    egrid = np.logspace(-1,3,num = bins)
    e_mid = (egrid[1:] - egrid[:-1])/(np.log10(egrid[1:])-np.log10(egrid[:-1]))
    e_mid = e_mid * 1.60218e-9 #convert to erg
    
    def integrate(pars):
        continuum = _models.lmodxillver(pars, egrid[:-1], egrid[1:])
        integral = (continuum*e_mid).sum()
        return integral
    
    with Parallel(n_jobs=20,verbose=0) as parallel:
        integrals = parallel(delayed(integrate)(pars) for pars in xill_pars)
    integrals = np.asarray(integrals)
    Anorm = F/(g_so**(gamma-2)*integrals)
    pars[:,19] = Anorm   #Replace flux generated with Anorm parameters
    return pars

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
    bad_disks = np.nonzero((lhc[:,6]>2.75)&(10**lhc[:,8]>4)&(lhc[:,9]>17))
    #check if the calculated hubble constant for the set of parameters
    #if outside of 60km/s/Mpc <= H0 <= 80km/s/Mpc
    hubble = lhc[:,5]*3e6/(10**lhc[:,7]*0.001)
    bad_dists = np.nonzero((hubble < 60) | (hubble > 80))
    heights = 1+ np.sqrt(1-lhc[:,1]**2)
    bad_heights = np.nonzero((lhc[:,0]<heights))
    #Check luminosities aren't super eddington or too small to see
    F = 10**lhc[:,19]       #Flux of corona in erg/cm^2/s
    D = 10**lhc[:,7]        #distance of objects
    D = 3.086e21 * D        #distance in cm
    M_solar = 10**lhc[:,13] #mass of the object
    L = 4*np.pi*(D**2)*F    #luminosity of corona in erg/cm^2/s
    Ledd = 1.26e38*M_solar  #eddington luminosity
    bad_Ls = np.nonzero((L > 1.05*Ledd)|(L < 1e-4*Ledd))
    #collates all bad sets together
    bad_sets = np.unique(np.concatenate((bad_disks,bad_heights,bad_dists,
                                         bad_Ls),axis=None))
    #removes all unphysical sets from the parameter sets
    new_lhc = np.delete(lhc,bad_sets,0)
    #convert fluxes to Anorm
    new_lhc = Anorm_wrapper(new_lhc)
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
    z_range = [0,0.1]
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

def lhc_1():
    """
    Generates valid ranges of parameters for spin only.

    Returns
    -------
    range_all : list
        gives parameter ranges for each of the given parameters listed. Used 
        in the latin hypercube sampling

    """
    r_inner_range = [np.log10(1),np.log10(400)]
    range_all = [r_inner_range]
    
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
    height_range = [np.log10(1.5),np.log10(100)]
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
    height_range = [np.log10(1.5),np.log10(100)]
    spin_range = [0,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    z_range = [0,0.1]
    Gamma_range = [1.4,3.4]
    distance_range = [np.log10(3.5e5),np.log10(5e7)]
    Afe_range = [np.log10(0.5),np.log10(10)]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    nH_range = [np.log10(1e-3),np.log10(200)]
    boost_range = [np.log10(1e-2),np.log10(10)]
    mass_range = [np.log10(1e4),np.log10(1e11)]
    honr_range = [0,0.176]
    b1_range = [0,2]
    b2_range = [-4,4]
    phiAB_range = [-3.14,3.14]
    g_range = [0,0.5]
    flux_range = [np.log10(1e-12),np.log10(1e-8)]
    
    
    range_all = [height_range,spin_range,inclination_range,r_inner_range,
                 r_outer_range,z_range,Gamma_range,distance_range,Afe_range,
                 logNe_range,kte_range,nH_range,boost_range,mass_range,
                 honr_range,b1_range,b2_range,phiAB_range,g_range,flux_range]
    
    return range_all

def lhc_10():
    """
    Limited form of lhc_range_gen that returns ranges for only a limited amount
    of parameters.

    Returns
    -------
    range_all : list
        a list of ranges of parameter spaces to generate from.

    """
    spin_range = [0.1,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    Gamma_range = [1.4,3.4]
    Afe_range = [np.log10(0.5),np.log10(10)]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    nH_range = [np.log10(1e-3),np.log10(200)]
    mass_range = [np.log10(1e4),np.log10(1e11)]
    
    range_all = [spin_range,inclination_range,r_inner_range,r_outer_range,
                 Gamma_range,Afe_range,logNe_range,kte_range,nH_range,
                 mass_range]
    
    return range_all

def lhc_5():
    """
    Limited form of lhc_range_gen that returns ranges for only a limited amount
    of parameters.

    Returns
    -------
    range_all : list
        a list of ranges of parameter spaces to generate from.

    """
    spin_range = [0.1,0.998]
    inclination_range = [np.log10(1),np.log10(80)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    mass_range = [np.log10(1e4),np.log10(1e11)]
    
    range_all = [spin_range,inclination_range,r_inner_range,r_outer_range,
                 mass_range]
    
    return range_all

def nn_pars_to_rtdist(nn_pars, ReIm, pars_list, negatives, logged):
    """
    Converts sampled paramters for neural network training into correct format
    for use in generating data and adds non-sampled parameters needed by the
    model. This is the generic form in which the list of sampled parameters 
    and manipulations are passed to the function

    Parameters
    ----------
    nn_pars : np.ndarray
        array of sampled parameters.
    ReIm : int
        value of ReIm from rtdist to determine the type of output rtdist 
        produces.
    pars_list : list
        list of indexes of parameters.
    negatives : list
        list of indexes of parameters that need to be turned positive.
    logged : list
        list of indexes of parameters that need to be transformed to power of
        10.

    Returns
    -------
    converted_pars : np.ndarray
        array of parameters to be used for parameter generation.

    """
    #set up base parameters which can be used to fix parameter sets to
    #reasonable values
    pars_base = [6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,0,
                 0,ReIm,0,-0.8,0.3,2.2e-4,1,1.]
    #set up base parameters to transform according to sampled parameters
    converted_pars = []
    for i in range(nn_pars.shape[0]):
        converted_pars.append(pars_base)
    converted_pars = np.asarray(converted_pars)
    #iterate through 
    for i, parameter in enumerate(pars_list):
        if parameter in logged:
            converted_pars[:,parameter] = 10**nn_pars[:,i]
        else:
            converted_pars[:,parameter] = nn_pars[:,i]
        if parameter in negatives:
            converted_pars[:,parameter] = -converted_pars[:,parameter]
    return converted_pars

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
   
    spin = np.linspace(0.1,0.998,size)
    inc = np.linspace(np.log10(1),np.log10(80),size)
    r_in = np.linspace(np.log10(1),np.log10(400),size)
    r_outer_range = np.linspace(np.log10(400),np.log10(1e5),size)
    mass = np.linspace(np.log10(1e4),np.log10(1e11),size)
    
    #create parameter grid
    theta_init = []
    for a in spin:
        for i in inc:
                for r_i in r_in:
                    for r in r_outer_range:
                        for m in mass:
                            theta_init.append([a,i,r_i,r,m])
    theta_init = np.asarray(theta_init)
    #convert to rtdist model compatible parameters
    pars_list = [1,2,3,4,13]
    negatives = [3]
    logged = [2,3,4,13]
    #generate physical models of test set
    theta_flux = nn_pars_to_rtdist(theta_init, 0, pars_list, negatives, logged)
    theta_lags = nn_pars_to_rtdist(theta_init, 6, pars_list, negatives, logged)
    
    with Parallel(n_jobs=20,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        flux = parallel(delayed(rtdist_flux)(pars, egrid)
                                        for pars in theta_flux)
        flux = np.array(flux)
        flux, theta_flux, theta_lags = spectraChecker(flux,theta_flux,theta_lags,
                                                      1e-11)
        lags_data_init = parallel(delayed(rtdist_lags)(pars, lags_egrid)
                                        for pars in theta_lags)
        lags = np.array(lags_data_init)
    
    idxs = np.arange(0,flux.shape[0])
    np.random.shuffle(idxs)
    tra_idx = idxs[:int(0.9*len(idxs))]
    tes_idx = idxs[int(0.9*len(idxs)):]
    
    #Splitting data and parameters into training and testing datasets
    train_flux = flux[tra_idx]
    train_lags = lags[tra_idx]
    train_flux_pars = theta_flux[tra_idx]
    train_lags_pars = theta_lags[tra_idx]
    
    test_flux = flux[tes_idx]
    test_lags = lags[tes_idx]
    test_flux_pars = theta_flux[tes_idx]
    test_lags_pars = theta_lags[tes_idx]
    
    print("Saving to disk")
    #save data for the first time in text files
    saveData(train_flux, train_flux_pars, 
             "data/locations/",f"loc_{fname}_flux.csv")
    saveData(train_lags, train_lags_pars, 
             "data/locations/",f"loc_{fname}_lags.csv")
    saveData(test_flux, test_flux_pars, 
             "data/locations/",f"loc_{fname}_flux_test.csv")
    saveData(test_lags, test_lags_pars, 
             "data/locations/",f"loc_{fname}_lags_test.csv")
    
    print("Performing data cleanup")
    readAndRemoveNans(f"data/locations/loc_{fname}_flux.csv", 
                      f"data/locations/loc_{fname}_lags.csv")
    readAndRemoveNans(f"data/locations/loc_{fname}_flux_test.csv", 
                      f"data/locations/loc_{fname}_lags_test.csv")
    return

def generate_flux_dists(AGN_name):
    wrk_dir = os.getcwd()
    rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
    rmf = unpack_rmf(rmf_name)
    egrid = rmf.e_min #energy grid used to evaluate the xspec model
    
    labels = ["height","a","inc","rin","rout","z","Gamma","Dkpc","Afe",
              "logNe","kte","nH","boost","mass","honr","b1","b2","phiAB","g",
              "Anorm"]
    range_AGN = np.asarray(lhc_AGN())
    
    #pre generate Latin Hypercube samples.
    theta_agn = lhc_generation(int(1e3), range_AGN)
    
    pars_list = [1,2,3,4,13]
    negatives = [3]
    logged = [2,3,4,13]
    #generate physical models of test set
    iter_agn = nn_pars_to_rtdist(theta_agn, 0, pars_list, negatives, logged)
    
    with Parallel(n_jobs=20,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        AGN_flux = parallel(delayed(rtdist_erg_flux)(pars, egrid)
                                        for pars in iter_agn)
        AGN_spec = parallel(delayed(rtdist_flux)(pars, egrid)
                                        for pars in iter_agn)
    AGN_flux = np.asarray(AGN_flux)
    AGN_spec = np.asarray(AGN_spec)
    
    saveData(AGN_spec,iter_agn, "data/flux/", f"{AGN_name}_range.csv")
    
    AGN = pd.DataFrame(data=theta_agn,columns=labels)
    AGN["flux"] = AGN_flux
    
    AGN.to_csv(f"data/flux/{AGN_name}.csv")
    return
    
def generate_test_set(size, egrid, lags_egrid, lhc_gen, limited=False):
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
    theta_lhc = lhc_generation(size, range_all,limited=limited)
    
    pars_list = [1,2,3,4,13]
    negatives = [3]
    logged = [2,3,4,13]
    
    pars_list = [1]
    negatives = []
    logged = []
    #generate physical models of test set
    theta_flux = nn_pars_to_rtdist(theta_lhc, 0, pars_list, negatives, logged)
    theta_lags = nn_pars_to_rtdist(theta_lhc, 6, pars_list, negatives, logged)
    
    with Parallel(n_jobs=20,verbose=5) as parallel:
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
                               lags_test_name):
    # compute the physical model for these thetas
    pars_list = [1,2,3,4,13]
    negatives = [3]
    logged = [2,3,4,13]
    #generate physical models of test set
    theta_flux = nn_pars_to_rtdist(theta_query, 0, pars_list, negatives, logged)
    theta_lags = nn_pars_to_rtdist(theta_query, 6, pars_list, negatives, logged)
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

def lhc_generation(size,range_all, limited = False):
    if limited == False:
        def lhc_cycle(size,range_all):
            sampler = scipy.stats.qmc.LatinHypercube(d=len(range_all))
            sample = sampler.random(n=size)
            theta_lhc = scipy.stats.qmc.scale(sample, range_all[:,0], range_all[:,1])
            final_lhc = lhc_filter(theta_lhc)
            return final_lhc
        if size > 1e6:
            gen_size = 1e6
        else:
            gen_size = size
        lhc = lhc_cycle(gen_size, range_all)
        percent = 0
        while lhc.shape[0] < size:
            if percent <= ((lhc.shape[0]/size)*100 - 10):
                print(f"Currently {lhc.shape[0]}/{size}.")
                percent = np.round((lhc.shape[0]/size)*100,0)/10
                percent = np.floor(percent)*10
            lhc_temp = lhc_cycle(gen_size, range_all)
            lhc = np.concatenate((lhc,lhc_temp),axis=0)
        np.random.shuffle(lhc)
        lhc = lhc[:size]
        return lhc
    else:
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
    
    pars_list = [1,2,3,4,13]
    negatives = [3]
    logged = [2,3,4,13]
    
    #generate physical models of test set
    theta_flux = nn_pars_to_rtdist(theta_init, 0, pars_list, negatives, logged)
    theta_lags = nn_pars_to_rtdist(theta_init, 6, pars_list, negatives, logged)
    print("Parallelized model generation")
    
    print("Generating flux models")
    flux =  Parallel(n_jobs=20,verbose=5)(delayed(rtdist_flux)(pars, egrid)
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
    lags_query =  Parallel(n_jobs=20,verbose=5)(delayed(rtdist_lags)(pars, lags_egrid)
                                    for pars in theta_lags)
    lags_query = np.asarray(lags_query)
    print("Saving lags data")
    saveData(lags_query, theta_lags,"data/locations/",lags_name, lags=True)
    del theta_flux, theta_lags, lags_query
    
    print("Performing data cleanup")
    readAndRemoveNans(f"data/locations/{flux_name}", 
                      f"data/locations/{lags_name}")
    return lhc_idx
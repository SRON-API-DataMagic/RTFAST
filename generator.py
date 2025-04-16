"""
This program deals with generating rtdist models with given parameters. For use
for generating sampels for training the emulator for rtdist.
"""
import numpy as np
import os
from joblib import Parallel, delayed
#from processing import saveData, spectraChecker
import scipy
import f2py_interface as ib
import matplotlib.pyplot as plt

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
    model = ib.reltransDCp(egrid, pars)
    return model

def lhc_filter_20(lhc):
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
    #removes parameter sets that have a photon index higher than 3 AND a iron
    #solar abundance above 6 AND a electron density in the disk of higher than
    #10^19.
    bad_disks = np.nonzero((lhc[:,6]>2.75)&(10**lhc[:,8]>4)&(lhc[:,9]>17))
    #coronas within the blackhole horizon
    heights = 1+ np.sqrt(1-lhc[:,1]**2)
    bad_heights = np.nonzero(10**lhc[:,0]<1.5*heights)
    #collates all bad sets together
    bad_sets = np.unique(np.concatenate((bad_disks,bad_heights),axis=None))
    #removes all unphysical sets from the parameter sets
    new_lhc = np.delete(lhc,bad_sets,0)
    return new_lhc

def lhc_ranges():
    """
    Generates valid ranges of parameters of AGN to be trained on
    
    Returns
    -------
    range_all : list
        gives parameter ranges for each of the given parameters listed. Used 
        in the latin hypercube sampling
    
    """
    height_range = [np.log10(1.5),np.log10(1e4)]
    spin_range = [-0.998,0.998]
    inclination_range = [np.log10(1),np.log10(89)]
    r_inner_range = [np.log10(1),np.log10(400)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    Gamma_range = [1.4,3.4]
    logxi_range = [0,4.7]
    Afe_range = [np.log10(0.5),np.log10(10)]
    logNe_range = [15,20]
    kte_range = [np.log10(5),np.log10(500)]
    
    
    range_all = [height_range,spin_range,inclination_range,r_inner_range,
                 r_outer_range,Gamma_range,logxi_range,Afe_range,
                 logNe_range,kte_range]
    
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
    pars_base = [6,0.9,57,-1,2e4,0,2.45,3,1,17,50.,0,0,3e6,0,0,0,
                 0,0,0,1,1.]
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


def lhc_generation(size,range_all,limited = False, lhc_filter = lhc_filter_20):
    if limited == False:
        def lhc_cycle(size,range_all):
            sampler = scipy.stats.qmc.LatinHypercube(d=len(range_all))
            sample = sampler.random(n=size)
            theta_lhc = scipy.stats.qmc.scale(sample, range_all[:,0], range_all[:,1])
            final_lhc = lhc_filter(theta_lhc)
            return final_lhc
        if size < 1e5:
            gen_size = int(1e5)
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

def generate_dataset(range_AGN,pars_list,negatives,logged,egrid):
    
    cpu_num = os.cpu_count()
    
    theta_lhc = lhc_generation(int(1000), range_AGN, limited=False, 
                                         lhc_filter=lhc_filter_20)
    
    #generate physical models of test set
    theta_flux = nn_pars_to_rtdist(theta_lhc, 0, pars_list, negatives, logged)
    print("Parallelized model generation")
    
    print("Generating flux models")
    
    with Parallel(n_jobs=cpu_num,verbose=1,backend="multiprocessing") as parallel:
        flux =  parallel(delayed(rtdist_flux)(pars,egrid)
                                        for pars in theta_flux)
        flux = np.asarray(flux)
    print("Checking for spectra below threshold")
    #flux, theta_flux = spectraChecker(flux,theta_flux,1e-11)
    
    plt.plot(egrid,flux,alpha=0.1,c="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Photons/cm^2/s/keV")
    plt.show()
    plt.close()
    
    plt.plot(egrid,(egrid**2)*flux,alpha=0.1,c="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (keV)")
    plt.ylabel("keV^2/cm^2/s/keV")
    plt.show()
    plt.close()

def main():
    range_AGN = np.asarray(lhc_ranges())
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,12,13,23]
    
    egrid = np.logspace(-1,2,num=1000)
    
    generate_dataset(range_AGN, pars_list, negatives, logged, egrid)
    
    
if __name__ == "__main__":
    main()
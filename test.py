"""
This is a quick training exercise to test if using PCA on spectra and using very
lightweight NNs is a viable alternative to what we've been doing up until now.
"""

from generator import rtdist_flux, nn_pars_to_rtdist
from generator import rtdist_lags, lhc_filter_20, lhc_generation, lhc_AGN
from training import grid_training_loop, train_flux, test_flux, PCALoss
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from sherpa.astro.io import read_arf
from processing import saveData, spectraChecker, mergeSaveData
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat

from dataStructures import PCADataset
from network import DynamicNetwork

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import corner

def generate_lags_from_parameters(egrid_lo,egrid_hi,fmin,fmax,ReIm=-1):
    """
    Generates lags from previously chosen parameters

    Parameters
    ----------
    ReIm : int
        Sets output of rtdist. The default is 3 (real parts of the cross
        spectrum).

    Returns
    -------
    None.

    """
    
    if ReIm == -1:
        cross_type = "real"
    elif ReIm == -2:
        cross_type = "imag"
    
    pars_val = pd.read_csv("data/locations/locs_20_spectra_val.csv")
    pars_tra = pd.read_csv("data/locations/locs_20_spectra_tra.csv")
    
    pars_val["ReIm"] = ReIm
    pars_tra["ReIm"] = ReIm
    
    pars_val["fmin"] = fmin
    pars_tra["fmin"] = fmin
    
    pars_val["fmax"] = fmax
    pars_tra["fmax"] = fmax
    
    print(pars_val)
    
    theta_val = np.asarray(pars_val)
    theta_tra = np.asarray(pars_tra)
    
    theta_val = theta_val[:,:-1]
    theta_tra = theta_tra[:,:-1]
    
    print(theta_val)
    cpu_num = os.cpu_count()
    
    load_size = int(1e5)
    
    no_loads_val = int(np.ceil(len(theta_val)/load_size))
    no_loads_tra = int(np.ceil(len(theta_tra)/load_size))
    
    print(f"Loading val in {no_loads_val} sets")
    print(f"Loading tra in {no_loads_tra} sets")
    
    
    with ProcessPoolExecutor(max_workers=20) as executor:
        """
        for i in range(3,4): #generate 4e6 datapoints
            print(f"Generating load {i}")
            if i != no_loads_val-1:
                pars_val = theta_val[i*load_size:(i+1)*load_size]
            else:
                pars_val = theta_val[i*load_size:]
            
            val = list(executor.map(rtdist_lags, pars_val,
                                    repeat(egrid_lo),repeat(egrid_hi)))
            val = np.asarray(val)
            
            print("Saving data")
            if i == 0:
                saveData(val, theta_val, 
                         "data/locations/",f"locs_20_{cross_type}_{fmin}_{fmax}_val.csv",lags=True)
            else:
                saveData(val, theta_val, 
                         "data/locations/","locs_temp.csv",lags=True)
                train = pd.read_csv("data/locations/locs_temp.csv")
                mergeSaveData(train, pd.read_csv("data/locations/locs_temp.csv"),
                              "data/locations/", f"locs_20_{cross_type}_{fmin}_{fmax}_val.csv")
        """
        for i in range(40): #generate 4e6 datapoints
            print(f"Generating load {i}")
            if i == no_loads_tra:
                break
            if i != no_loads_tra-1:
                pars_tra = theta_tra[i*load_size:(i+1)*load_size]
            else:
                pars_tra = theta_tra[i*load_size:]
            
            tra = list(executor.map(rtdist_lags, pars_tra,
                                    repeat(egrid_lo),repeat(egrid_hi)))
            tra = np.asarray(tra)
        
            print("Saving data")
            if i == 0:
                saveData(tra, theta_tra, 
                         "data/locations/",f"locs_20_{cross_type}_{fmin}_{fmax}_tra.csv",lags=True)
            else:
                saveData(tra, theta_tra, 
                         "data/locations/","locs_temp.csv",lags=True)
                train = pd.read_csv("data/locations/locs_temp.csv")
                mergeSaveData(train, pd.read_csv("data/locations/locs_temp.csv"),
                              "data/locations/", f"locs_20_{cross_type}_{fmin}_{fmax}_tra.csv")

def new_set(range_AGN,pars_list,negatives,logged,egrid_lo,egrid_hi):
    
    cpu_num = os.cpu_count()
    
    theta_lhc = lhc_generation(int(1e6), range_AGN, limited=False, 
                                         lhc_filter=lhc_filter_20)
    labels = ["height","a","inc","rin","rout","z","Gamma","Dkpc","Afe","logNe","kte",
              "nH","boost","mass","honr","b1","b2","phiAB","g","Anorm"]
    figure = corner.corner(
        theta_lhc,
        labels=labels,
        )
    plt.savefig("loss/parameter_dists.png")
    plt.close()
    
    #generate physical models of test set
    theta_flux = nn_pars_to_rtdist(theta_lhc, 0, pars_list, negatives, logged)
    theta_lags = nn_pars_to_rtdist(theta_lhc, 6, pars_list, negatives, logged)
    print("Parallelized model generation")
    
    print("Generating flux models")
    
    with Parallel(n_jobs=cpu_num,verbose=1,backend="multiprocessing") as parallel:
        flux =  parallel(delayed(rtdist_flux)(pars,egrid_lo,egrid_hi)
                                        for pars in theta_flux)
        flux = np.asarray(flux)
    print("Checking for spectra below threshold")
    flux, theta_flux, theta_lags = spectraChecker(flux,theta_flux,theta_lags,
                                                  1e-11)
    
    idxs = np.arange(0,flux.shape[0])
    np.random.shuffle(idxs)
    tra_idx = idxs[:int(0.9*len(idxs))]
    val_idx = idxs[int(0.9*len(idxs)):]
    
    #Splitting data and parameters into training and valting datasets
    train_flux_data = flux[tra_idx]
    train_flux_pars = theta_flux[tra_idx]
    
    val_flux_data = flux[val_idx]
    val_flux_pars = theta_flux[val_idx]
    
    print("Saving flux data")
    saveData(train_flux_data, train_flux_pars, 
             "data/locations/","PCA_locs_flux_temp.csv")
    saveData(val_flux_data, val_flux_pars, 
             "data/locations/","PCA_locs_flux_val.csv")

def merge():
    train = pd.read_csv("data/locations/PCA_locs_flux_temp.csv")
    val = pd.read_csv("data/locations/PCA_locs_flux_val.csv")
    
    train_name = "locs_20_spectra_tra.csv"
    val_name = "locs_20_spectra_val.csv"
    #save final curated datasets back to disk for use
    mergeSaveData(train, pd.read_csv(f"data/locations/{train_name}"),
                  "data/locations/", train_name)
    mergeSaveData(val, pd.read_csv(f"data/locations/{val_name}"),
                  "data/locations/",val_name)

def main():
    wrk_dir = os.getcwd()
    
    arf_name = wrk_dir+"/ResponseFiles/PN.arf"
    arf = read_arf(arf_name)
    egrid_lo,egrid_hi = arf.energ_lo[arf.energ_lo>0.1],arf.energ_hi[arf.energ_lo>0.1]
    
    #set envionmental variables required in xspec with simrtdist
    environ_vars = {"REV_VERB":"0","MU_ZONES":"1","ION_ZONES":"1","A_DENSITY":"1",
                    "EMIN_REF":"0.5","EMAX_REF":"10","EMIN_REF2":"0.5",
                    "EMAX_REF2":"10", "SEED_SIM":"-2851043",
                    "RMF_SET":wrk_dir+"/ResponseFiles/PN.rmf",
                    "ARF_SET":wrk_dir+"/ResponseFiles/PN.arf",
                    "BKG_SET":wrk_dir+"/ResponseFiles/PNbackground_spectrum.fits",
                    "BACKSCL":"1.0"}
    
    for key in environ_vars:
        os.environ[key] = environ_vars[key]
    
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13]
    #note that Anorm should be added to logged when training - Anorm wrapper
    #calculates the real value of Anorm rather than the logarithm
    
    range_AGN = np.asarray(lhc_AGN())
    num_pars = len(pars_list)
    print(num_pars)
    """
    for i in range(10):
        print(f"loop {i}")
        new_set(range_AGN,pars_list,negatives,logged,egrid_lo,egrid_hi)
        print("Successfully saved")
        merge()
        print("Successfully merged")
    """
    fmins = [5e-5,1e-4,5e-3]
    fmaxs = [1e-4,5e-3,1e-2]
    for fmin, fmax in zip(fmins,fmaxs):
        generate_lags_from_parameters(egrid_lo,egrid_hi,fmin,fmax,ReIm=-1)
        generate_lags_from_parameters(egrid_lo,egrid_hi,fmin,fmax,ReIm=-2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,23] #Anorm is relogged when actually training
    
    val_dataset = PCADataset("data/locations/locs_20_spectra_val.csv",
                               pars_list,negatives,logged,scale_bool = False,
                               PCA_loc="scalers/PCA_20_spec.bin",
                               comp_loc="scalers/comp_20_spec.bin",
                               spec_scal_loc="scalers/spec_20_spec.bin")
    val_loader = DataLoader(val_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    
    train_dataset = PCADataset("data/locations/locs_20_spectra_tra.csv",
                               pars_list,negatives,logged,scale_bool = False,
                               PCA_loc="scalers/PCA_20_spec.bin",
                               comp_loc="scalers/comp_20_spec.bin",
                               spec_scal_loc="scalers/spec_20_spec.bin")
    tra_loader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    loss_fn = PCALoss(val_dataset.pca.explained_variance_ratio_, device)
    
    model = DynamicNetwork(20,40,8,256,"GELU")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    train = train_flux
    test =  test_flux
    
    grid_training_loop(model, optimizer, train, test, tra_loader, 
                       val_loader, loss_fn, device, "20_pars", "flux", 
                       epochs = 1000)
    

if __name__ == "__main__":
    main()
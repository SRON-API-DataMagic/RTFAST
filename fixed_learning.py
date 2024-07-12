"""
This is a quick training exercise to test if using PCA on spectra and using very
lightweight NNs is a viable alternative to what we've been doing up until now.
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sherpa.astro.io import read_arf
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import corner
import logging
import traceback

from dataStructures import PCADataset
from network import DynamicNetwork
from processing import saveData, spectraChecker, mergeSaveData
from generator import rtdist_flux, nn_pars_to_rtdist
from generator import rtdist_lags, lhc_filter_20, lhc_generation
from training import grid_training_loop, train_flux, test_flux, PCALoss

def generate_lags_from_parameters(egrid_lo,egrid_hi,fmin,fmax,ReIm=-1,
                                  start = False):
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
    
    theta_val = np.asarray(pars_val)
    theta_tra = np.asarray(pars_tra)
    
    theta_val = theta_val[:,:-1]
    theta_tra = theta_tra[:,:-1]
    
    cpu_num = os.cpu_count()
    
    load_size = int(1e5)
    
    no_loads_val = int(np.ceil(len(theta_val)/load_size))
    no_loads_tra = int(np.ceil(len(theta_tra)/load_size))
    
    print(f"Loading val in {no_loads_val} sets")
    print(f"Loading tra in {no_loads_tra} sets")
    
    try:
        with ProcessPoolExecutor(max_workers=cpu_num) as executor:
            #validation set
            for i in range(4): #generate 4e6 datapoints
                print(f"Generating load {i}")
                pars_val = theta_val[i*load_size:(i+1)*load_size]
                
                val = list(executor.map(rtdist_lags, pars_val,
                                        repeat(egrid_lo),repeat(egrid_hi)))
                val = np.asarray(val)
                
                print("Saving data")
                if i == 0 and start == True:
                    saveData(val, theta_val, 
                             "data/locations/",f"locs_20_{cross_type}_val.csv",lags=True)
                else:
                    saveData(val, theta_val, 
                             "data/locations/","locs_temp.csv",lags=True)
                    train = pd.read_csv("data/locations/locs_temp.csv")
                    mergeSaveData(train, pd.read_csv(f"data/locations/locs_20_{cross_type}_tra.csv"),
                                  "data/locations/", f"locs_20_{cross_type}_val.csv")
            #training set
            for i in range(4,44): #generate 4e6 datapoints
                print(f"Generating load {i}")
                pars_tra = theta_val[i*load_size:(i+1)*load_size]
                
                tra = list(executor.map(rtdist_lags, pars_tra,
                                        repeat(egrid_lo),repeat(egrid_hi)))
                tra = np.asarray(tra)
            
                print("Saving data")
                if i == 0 and start == True:
                    saveData(tra, theta_tra, 
                             "data/locations/",f"locs_20_{cross_type}_tra.csv",lags=True)
                else:
                    saveData(tra, theta_tra, 
                             "data/locations/","locs_temp.csv",lags=True)
                    train = pd.read_csv("data/locations/locs_temp.csv")
                    mergeSaveData(train, pd.read_csv(f"data/locations/locs_20_{cross_type}_tra.csv"),
                                  "data/locations/", f"locs_20_{cross_type}_tra.csv")
    except Exception as e:
        logging.error(f"Exception in main process: {e}\n{traceback.format_exc()}")
        
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,23]
    
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
    
    train = train_flux
    test =  test_flux
    
    for i in range(10):
        print(f"Training model {i+1}")
        model = DynamicNetwork(20,40,8,256,"GELU")
        model.to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)
        
        grid_training_loop(model, optimizer, train, test, tra_loader, 
                           val_loader, loss_fn, device, f"{i}_20_pars", "flux", 
                           epochs = 2000)
    

if __name__ == "__main__":
    main()

"""
This is a quick training exercise to test if using PCA on spectra and using very
lightweight NNs is a viable alternative to what we've been doing up until now.
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataStructures import PCADataset
from network import DynamicNetwork, DynamicResNetwork
from training import grid_training_loop, train_flux, test_flux, PCALoss
from sherpa.astro.io import read_arf
from generator import lhc_AGN, lhc_generation, lhc_filter_20, nn_pars_to_rtdist
from generator import rtdist_flux
from processing import saveData, spectraChecker, mergeSaveData

def new_set(range_AGN,pars_list,negatives,logged,egrid_lo,egrid_hi):
    
    cpu_num = os.cpu_count()
    
    theta_lhc = lhc_generation(int(1e5), range_AGN, limited=False, 
                                         lhc_filter=lhc_filter_20)
    
    #generate physical models of test set
    theta_flux = nn_pars_to_rtdist(theta_lhc, 0, pars_list, negatives, logged)
    print("Parallelized model generation")
    
    print("Generating flux models")
    
    with Parallel(n_jobs=cpu_num,verbose=1,backend="multiprocessing") as parallel:
        flux =  parallel(delayed(rtdist_flux)(pars,egrid_lo,egrid_hi)
                                        for pars in theta_flux)
        flux = np.asarray(flux)
    print("Checking for spectra below threshold")
    flux, theta_flux = spectraChecker(flux,theta_flux,1e-11)
    
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
    """
    wrk_dir = os.getcwd()
    arf_name = wrk_dir+"/ResponseFiles/PN.arf"
    arf = read_arf(arf_name)
    egrid_lo,egrid_hi = arf.energ_lo[arf.energ_lo>0.1],arf.energ_hi[arf.energ_lo>0.1]
    range_AGN = np.asarray(lhc_AGN())
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,12,13]
    
    for i in range(100):
        new_set(range_AGN, pars_list, negatives, logged, egrid_lo, egrid_hi)
        merge()
    
    exit()
    """
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,12,13,23]
    
    val_dataset = PCADataset("data/locations/locs_PCA_comps_val.csv",
                               pars_list,negatives,logged,scale_bool = False,
                               PCA_loc="scalers/PCA_spec.bin",
                               comp_loc="scalers/comp_spec.bin",
                               spec_scal_loc="scalers/spec_spec.bin",
                               load_pca=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    comps = val_dataset.pca.n_components
    train_dataset = PCADataset("data/locations/locs_PCA_comps_tra.csv",
                               pars_list,negatives,logged,scale_bool = False,
                               PCA_loc="scalers/PCA_spec.bin",
                               comp_loc="scalers/comp_spec.bin",
                               spec_scal_loc="scalers/spec_spec.bin",
                               load_pca=True)
    tra_loader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    
    loss_fn = PCALoss(val_dataset.pca.explained_variance_ratio_, device)
    
    train = train_flux
    test =  test_flux
    
    for i in range(10):
        print(f"Training model {i+1}")
        model = DynamicNetwork(17,comps,12,256)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=1e-4)
        
        grid_training_loop(model, optimizer, train, test, tra_loader, 
                           val_loader, loss_fn, device, f"{i}_20_pars", "flux", 
                           epochs = 2000)
    

if __name__ == "__main__":
    main()

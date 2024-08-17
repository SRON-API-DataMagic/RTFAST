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
        if i == 0:
            renameData(pd.read_csv("data/locations/PCA_locs_flux_temp.csv"),
                       "data/locations/locs_20_spectra_tra.csv")
            renameData(pd.read_csv("data/locations/PCA_locs_flux_val.csv"),
                       "data/locations/locs_20_spectra_val.csv")
        else:
            merge()
    """
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,12,13,23]
    
    val_dataset = PCADataset("data/locations/locs_20_spectra_val.csv",
                               pars_list,negatives,logged,scale_bool = False,
                               PCA_loc="scalers/PCA_spec.bin",
                               comp_loc="scalers/comp_spec.bin",
                               spec_scal_loc="scalers/spec_spec.bin")
    val_loader = DataLoader(val_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    comps = val_dataset.pca.n_components
    train_dataset = PCADataset("data/locations/locs_20_spectra_tra.csv",
                               pars_list,negatives,logged,scale_bool = False,
                               PCA_loc="scalers/PCA_spec.bin",
                               comp_loc="scalers/comp_spec.bin",
                               spec_scal_loc="scalers/spec_spec.bin")
    tra_loader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    
    loss_fn = PCALoss(val_dataset.pca.explained_variance_ratio_, device)
    
    train = train_flux
    test =  test_flux
    
    for i in range(10):
        print(f"Training model {i+1}")
        model = DynamicNetwork(17,comps,8,512)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=5e-4)
        
        grid_training_loop(model, optimizer, train, test, tra_loader, 
                           val_loader, loss_fn, device, f"{i}_20_pars", "flux", 
                           epochs = 2000)
    

if __name__ == "__main__":
    main()

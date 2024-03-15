"""
This is a quick training exercise to test if using PCA on spectra and using very
lightweight NNs is a viable alternative to what we've been doing up until now.
"""

from generator import intialize_dataset, rtdist_flux, nn_pars_to_rtdist
import generator
from training import grid_training_loop, train_flux, test_flux, PCALoss
from training import bottleneck_training_loop
import numpy as np
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sherpa.astro.ui import unpack_rmf
from sherpa.astro.io import read_arf
from processing import saveData, spectraChecker, mergeSaveData
from joblib import Parallel, delayed, dump, load

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from dataStructures import PCADataset
from network import PCAFluxNetwork, PCANetwork, DynamicNetwork

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
import corner

import wandb

def new_set(range_AGN,pars_list,negatives,logged,egrid_lo,egrid_hi):
    
    theta_lhc = generator.lhc_generation(int(5e5), range_AGN, limited=False, 
                                         lhc_filter=generator.lhc_filter_20)
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
    theta_lags = nn_pars_to_rtdist(theta_lhc, 0, pars_list, negatives, logged)
    print("Parallelized model generation")
    
    print("Generating flux models")
    flux =  Parallel(n_jobs=20,verbose=5,backend="multiprocessing")(delayed(rtdist_flux)(pars,egrid_lo,egrid_hi)
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
             "data/locations/","locs_20_spectra_tra.csv")
    saveData(val_flux_data, val_flux_pars, 
             "data/locations/","locs_20_spectra_val.csv")

def merge():
    train = pd.read_csv("data/locations/PCA_locs_flux_temp.csv")
    test = pd.read_csv("data/locations/PCA_locs_flux_test_temp.csv")
    
    train_name = "PCA_locs_flux.csv"
    test_name = "PCA_locs_flux_test.csv"
    #save final curated datasets back to disk for use
    mergeSaveData(train, pd.read_csv(f"data/locations/{train_name}"),
                  "data/locations/", train_name)
    mergeSaveData(test, pd.read_csv(f"data/locations/{test_name}"),
                  "data/locations/",test_name)

def build_optimizer(model,optimizer_name,learning_rate):
    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(),lr=learning_rate)
    elif optimizer_name == "adamW":
        optimizer = AdamW(model.parameters(),lr=learning_rate)
    
    return optimizer

def wandb_sweep():
    config = wandb.config
    # Initialize a new wandb run
    name = f"{config.optimizer}_{config.num_layers}_{config.nodes}_{config.learning_rate}_{config.activation}"
    
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,23]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    val_dataset = PCADataset("data/locations/locs_20_spectra_val.csv",
                               pars_list,negatives,logged,scale_bool = True,
                               comps=40,
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

    tra_loader,val_loader = config.tra_loader,config.val_loader
    loss_fn,device = config.loss_fn,config.device
    
    model = DynamicNetwork(20, 40,
                           config.num_layers,config.nodes,
                           config.activation)
    model.to(device)
    torch.save(model.state_dict(), f"models/{name}.pth")
    run.log_model(path=f"models/{name}.pt", name=f"{name}")
    optimizer = build_optimizer(model, config.optimizer, 
                                config.learning_rate)
    
    loss_arr = []
    for epoch in range(config.epochs):
        (model,optimizer,
         train_loss,med_loss,std_loss) = train_flux(tra_loader,model,
                                             optimizer, loss_fn, device)
        loss = test_flux(val_loader, model, loss_fn, device)
        loss_arr.append(loss)
        wandb.log({"loss": loss,"med_loss": med_loss,"std_loss":std_loss,
                   "epoch": epoch}) 
        if loss == np.min(loss_arr):
            torch.save(model.state_dict(), f"models/{name}.pth")

def sweep_call():
    wandb_sweep()
    return

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
    
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,23]
    
    """
    pars_list = [1,2,3,6,7,8,9,11,13,23]
    negatives = [3]
    logged = [2,3,7,8,11,13,23]
    """
    """
    pars_list = [1,2,3,4,13]
    negatives = [3]
    logged = [2,3,4,13]
    """
    """
    pars_list = [3]
    negatives = [3]
    logged = [3]
    """
    range_AGN = np.asarray(generator.lhc_AGN())
    num_pars = len(pars_list)
    print(num_pars)
    
    #new_set(range_AGN,pars_list,negatives,logged,egrid_lo,egrid_hi)
    #merge()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    sweep_config = {
    'method': 'grid'
    }
    metric = {'name':'weighted_loss',
               'goal':'minimize'}
    sweep_config['metric'] = metric
    
    parameters_dict = {
    'optimizer': {
        'values': ['adam', 'adamW']
        },
    'num_layers': {
        'values': [4,6,8]
        },
    'nodes': {
        'values': [256,512,1024]
        },
    'epochs': {
          'values': [1500]
        },
    'learning_rate': {
        'values': [1e-4,5e-3,1e-3]
        },
    'activation': {
        'values': ["ReLU","GELU"]
        }
    }
    sweep_config['parameters'] = parameters_dict
    
    sweep_id = wandb.sweep(sweep_config, project="rtdist-emulator")
    
    wandb.agent(sweep_id=sweep_id, function=sweep_call)
    """
    model = PCANetwork(num_pars, val_dataset.data.shape[1])
    
    model.to(device)
    
    model.float()
    
    optimizer = Adam(model.parameters(),lr = 1e-3)
    train = train_flux
    test =  test_flux
    
    grid_training_loop(model, optimizer, train, test, train_dataloader, 
                       val_dataloader, loss_fn, device, "20_pars", "flux", 
                       epochs = 2000)
    """

if __name__ == "__main__":
    main()
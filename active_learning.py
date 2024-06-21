"""
This is the main program that trains the neural network.
"""
import numpy as np
import os

from sherpa.astro.ui import unpack_rmf

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd

from dataStructures import PCADataset
import network

from training import train_flux, test_flux
from training import active_training_loop
from training import PCALoss
from training import QBDC

def active_learning(device, wrk_dir):
    """
    Core method that collates together methods from other files to perform
    active learning based training by the query by dropout committee technique.
    Mostly passes information between different methods as well as setting
    core parameters (such as number of active learning loops or the energy
    grid being trained on). This method can also continue the training of a 
    previous model.

    Parameters
    ----------
    wrk_dir : string
        pass in the working directory of where you wish to work.
    device : string, optional
        This tells each method which cuda device to use. The default is cpu.

    Returns
    -------
    None.

    """
    print("Training using query by dropout committee")
    rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
    rmf = unpack_rmf(rmf_name)
    egrid = rmf.e_min #energy grid used to evaluate the xspec model
    egrid = egrid[egrid>0.1]
    
    active_loops = 75
    
    labels = ["height","a","inc","rin","rout","z","Gamma","distance","Afe","logNe","kte",
              "nH","boost","mass","honr","b1","b2","phiAB","g","Anorm"]
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,23]
    
    name = "locs_active.csv"
    val_name = "locs_active_val.csv"
    
    pool_csv = pd.read_csv("data/locations/locs_20_spectra_tra.csv")
    
    #select first 5000 spectra
    starting_csv = pool_csv.iloc[1000:100000]
    val_csv = pool_csv.iloc[:1000]
    starting_csv.to_csv("data/locations/locs_active_tra.csv",index=False)
    val_csv.to_csv("data/locations/locs_active_val.csv",index=False)
    print("Loaded pool and established starting training and validation sets")
    
    #remove first 5000 spectra from pool
    pool_csv = pool_csv.iloc[100000:]
    
    last_sig_tr = 1e7 #last significant best training loss (set large initially)
    last_sig_te = 1e7 #last significant best testing loss (set large initially)
    
    tr_loss_arr = []
    te_loss_arr = []
    
    loop_epochs = []
    
    active_loop_num = 0
    
    #create initial dataset object to create scaler
    flux_dataset = PCADataset("data/locations/locs_active_val.csv",
                              pars_list,negatives,logged,scale_bool = False,
                              PCA_loc="scalers/PCA_20_spec.bin",
                              comp_loc="scalers/comp_20_spec.bin",
                              spec_scal_loc="scalers/spec_20_spec.bin")
    
    loss_fn = PCALoss(flux_dataset.pca.explained_variance_ratio_, device)
    
    model = network.DynamicDropoutNetwork(20,40,8,256,"GELU")
    model.to(device)
    
    best_model = network.DynamicDropoutNetwork(20,40,8,256,"GELU")
    best_model.to(device)
    
    optimizer = Adam(model.parameters(),lr = 1e-4)
    
    print("Beginning training")
    while active_loop_num <= active_loops:
        pool_csv,val_csv = QBDC(name,val_name,active_loop_num,pool_csv,val_csv,
                                model,device,labels)
        
        if active_loop_num == 0:
            print("Creating dataloaders...")
            val_dataset = PCADataset("data/locations/locs_active_val.csv",
                                       pars_list,negatives,logged,scale_bool = False,
                                       PCA_loc="scalers/PCA_20_spec.bin",
                                       comp_loc="scalers/comp_20_spec.bin",
                                       spec_scal_loc="scalers/spec_20_spec.bin")
            val_loader = DataLoader(val_dataset, batch_size=1024, num_workers = 4, 
                                          shuffle=True)
            
            train_dataset = PCADataset("data/locations/locs_active_tra.csv",
                                       pars_list,negatives,logged,scale_bool = False,
                                       PCA_loc="scalers/PCA_20_spec.bin",
                                       comp_loc="scalers/comp_20_spec.bin",
                                       spec_scal_loc="scalers/spec_20_spec.bin")
            tra_loader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                                          shuffle=True)
        else:
            print("Loading new validation dataset")
            val_dataset = PCADataset("data/locations/locs_active_val.csv",
                                       pars_list,negatives,logged,scale_bool = False,
                                       PCA_loc="scalers/PCA_20_spec.bin",
                                       comp_loc="scalers/comp_20_spec.bin",
                                       spec_scal_loc="scalers/spec_20_spec.bin")
            val_loader = DataLoader(val_dataset, batch_size=1024, num_workers = 4, 
                                          shuffle=True)
            print("Adding new training data")
            train_dataset.add_data("data/locations/locs_active_.csv")
            tra_loader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                                          shuffle=True)
        
        print("Flux dataloaders created")
        #Train the flux model first
        (model, best_model, optimizer, loop_epochs, 
                te_loss_arr, tr_loss_arr, 
                last_sig_tr, last_sig_te) = active_training_loop(model, tra_loader, 
                                                          optimizer, loss_fn, 
                                                          device, val_loader, 
                                                          te_loss_arr, tr_loss_arr, 
                                                          last_sig_te, last_sig_tr, 
                                                          active_loop_num, loop_epochs, 
                                                          best_model, 
                                                          train_flux, test_flux,
                                                          mode = "flux",
                                                          stopping=20)
        #iterate loop number by 1
        active_loop_num += 1
        
    print("Completed training")
    print("Final best flux training loss:", last_sig_tr)
    print("Final best flux testing loss:", last_sig_te)
    torch.save(model.state_dict(), f"models/{name}_flux_final.pth")
    print(f"Saved PyTorch Model State to models/{name}_flux_final.pth")
    
    tr_loss_arr = np.asarray(tr_loss_arr)
    te_loss_arr = np.asarray(te_loss_arr)
    
    np.savetxt("loss/active_flux_te_loss.txt",te_loss_arr)
    np.savetxt("loss/active_flux_tr_loss.txt",tr_loss_arr)
    np.savetxt("loss/active_flux_epochs.txt",loop_epochs)

def main():
    wrk_dir = os.getcwd()
    print("Cuda is available:",torch.cuda.is_available())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    active_learning(device,wrk_dir)
    
if __name__ == "__main__":
    main()
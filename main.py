"""
This is the main program that trains the neural network.
"""
import numpy as np
import os

from sherpa.astro.ui import unpack_rmf

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from joblib import Parallel

from dataStructures import FluxData, LagsData, PCADataset, PCALagsDataset

import generator
import network

from training import train_flux, train_lags, test_flux, test_lags
from training import active_training_loop, grid_training_loop
from training import FluxLoss, LagLoss, PCALoss
from training import QBDC
from generator import intialize_dataset

def active_learning(device, wrk_dir, name):
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
    lags_egrid = np.logspace(np.log10(0.5),np.log10(11),num=26)
    
    active_loops = 30
    range_AGN = np.asarray(generator.lhc_trimmed_gen())
    
    labels = ["height","a","inc","rin","rout","z","Gamma","distance","Afe","logNe","kte",
              "nH","boost","mass","honr","b1","b2","phiAB","g","Anorm"]
    labels = ["a","inc","rin","distance","mass"]
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,23]
    
    pars_list = [1,2,3,4,13]
    negatives = [3]
    logged = [2,3,4,13]
    
    lhc_idx = 0
    theta_lhc = generator.lhc_generation(int(1e6), range_AGN, limited = True)
    
    flux_name = f"{name}_locs_flux.csv"
    flux_test_name = f"{name}_test_locs_flux.csv"
    flux_PCA = f"scalers/{name}_PCA_flux.bin"
    flux_comp = f"scalers/{name}_comp_flux.bin"
    
    lags_name = f"{name}_locs_lags.csv"
    lags_test_name = f"{name}_test_locs_lags.csv"
    lags_PCA = f"scalers/{name}_PCA_lags.bin"
    lags_comp = f"scalers/{name}_comp_lags.bin"
    
    num_pars = range_AGN.shape[0]
    
    lhc_idx = intialize_dataset(theta_lhc, egrid, lags_egrid, flux_name, 
                                lags_name, trimmed = True)
    
    last_sig_flux_tr = 1e7 #last significant best training loss (set large initially)
    last_sig_flux_te = 1e7 #last significant best testing loss (set large initially)
    last_sig_lags_tr = 1e7 #last significant best training loss (set large initially)
    last_sig_lags_te = 1e7 #last significant best testing loss (set large initially)
    
    flux_tr_loss_arr = []
    flux_te_loss_arr = []
    lags_tr_loss_arr = []
    lags_te_loss_arr = []
    
    loop_flux_epochs = []
    loop_lags_epochs = []
    
    active_loop_num = 0
    
    #create initial dataset object to create scaler
    flux_dataset = PCADataset(f"data/locations/{flux_name}",
                               pars_list,negatives,logged,
                               PCA_loc = flux_PCA,
                               comp_loc= flux_comp)
    lags_dataset = PCALagsDataset(f"data/locations/{lags_name}",
                               pars_list,negatives,logged,
                               PCA_loc = lags_PCA,
                               comp_loc= lags_comp)
    
    loss_fn_flux = PCALoss(flux_dataset.pca.explained_variance_ratio_,device)
    loss_fn_lags = PCALoss(lags_dataset.pca.explained_variance_ratio_,device)
    
    flux_model = network.PCAFluxNetwork(num_pars,flux_dataset.data.shape[1])
    flux_model.to(device)
    lags_model = network.PCAFluxNetwork(num_pars,lags_dataset.data.shape[1])
    lags_model.to(device)
    
    best_flux_model = network.HeavyFluxNetwork(num_pars,flux_dataset.data.shape[1])
    best_flux_model.to(device)
    best_lags_model = network.HeavyLagsNetwork(num_pars,lags_dataset.data.shape[1])
    best_lags_model.to(device)
    
    optimizer_flux = Adam(flux_model.parameters(),lr = 1e-3)
    optimizer_lags = Adam(lags_model.parameters(),lr = 1e-3)
    
    batch_size = 1024
    num_workers = 4
    
    print("Beginning training")
    with Parallel(n_jobs=20,verbose=3) as parallel:
        while active_loop_num <= active_loops:
            theta_lhc, lhc_idx = QBDC(flux_name, flux_test_name, lags_name, 
                                      lags_test_name, active_loop_num, 
                                      theta_lhc, lhc_idx, egrid, lags_egrid, 
                                      flux_model, lags_model, device, 
                                      labels, parallel)
            
            print("Creating dataloaders...")
            Xquery = PCADataset(f"data/locations/{flux_name}",
                                pars_list,negatives,logged,
                                scale_bool = False,
                                PCA_loc = flux_PCA,
                                comp_loc= flux_comp)
            flux_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            
            Xtest = PCADataset(f"data/locations/{flux_test_name}",
                                pars_list,negatives,logged,
                                scale_bool = False,
                                PCA_loc = flux_PCA,
                                comp_loc= flux_comp)
            flux_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                         num_workers = num_workers, shuffle=True)
            
            print("Flux dataloaders created")
            
            Xquery = PCALagsDataset(f"data/locations/{lags_name}",
                                    pars_list,negatives,logged,
                                    scale_bool=False,
                                    PCA_loc = lags_PCA,
                                    comp_loc= lags_comp)
            lags_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            
            Xtest = PCALagsDataset(f"data/locations/{lags_test_name}",
                                    pars_list,negatives,logged,
                                    scale_bool=False,
                                    PCA_loc = lags_PCA,
                                    comp_loc= lags_comp)
            lags_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                         num_workers = num_workers, shuffle=True)
            
            print("Lags dataloaders created")
            #Train the flux model first
            (flux_model, best_flux_model, optimizer_flux, loop_flux_epochs, 
                    flux_te_loss_arr, flux_tr_loss_arr, 
                    last_sig_flux_tr, last_sig_flux_te) = active_training_loop(flux_model, flux_dataloader, 
                                                              optimizer_flux, loss_fn_flux, 
                                                              device, flux_test_dataloader, 
                                                              flux_te_loss_arr, flux_tr_loss_arr, 
                                                              last_sig_flux_te, last_sig_flux_tr, 
                                                              active_loop_num, loop_flux_epochs, 
                                                              best_flux_model, 
                                                              train_flux, test_flux,
                                                              mode = "flux")
            #train the lags model second
            (lags_model, best_lags_model, optimizer_lags, loop_lags_epochs, 
                    lags_te_loss_arr, lags_tr_loss_arr, 
                    last_sig_lags_tr, last_sig_lags_te) = active_training_loop(lags_model, lags_dataloader, 
                                                              optimizer_lags, loss_fn_lags, 
                                                              device, lags_test_dataloader, 
                                                              lags_te_loss_arr, lags_tr_loss_arr, 
                                                              last_sig_lags_te, last_sig_lags_tr, 
                                                              active_loop_num, loop_lags_epochs, 
                                                              best_lags_model, 
                                                              train_flux, test_flux,
                                                              mode = "lags")
            #iterate loop number by 1
            active_loop_num += 1
            
        print("Completed training")
        print("Final best flux training loss:", last_sig_flux_tr)
        print("Final best flux testing loss:", last_sig_flux_te)
        torch.save(flux_model.state_dict(), f"models/{name}_flux_final.pth")
        print(f"Saved PyTorch Model State to models/{name}_flux_final.pth")
        
        flux_tr_loss_arr = np.asarray(flux_tr_loss_arr)
        flux_te_loss_arr = np.asarray(flux_te_loss_arr)
        
        np.savetxt(f"loss/{name}_flux_te_loss.txt",flux_te_loss_arr)
        np.savetxt(f"loss/{name}_flux_tr_loss.txt",flux_tr_loss_arr)
        np.savetxt(f"loss/{name}_flux_epochs.txt",loop_flux_epochs)
        
        print("Final best lags training loss:", last_sig_lags_tr)
        print("Final best lags testing loss:", last_sig_lags_te)
        torch.save(lags_model.state_dict(), f"models/{name}_lags_final.pth")
        print(f"Saved PyTorch Model State to models/{name}_lags_final.pth")
        
        lags_tr_loss_arr = np.asarray(lags_tr_loss_arr)
        lags_te_loss_arr = np.asarray(lags_te_loss_arr)
        
        np.savetxt(f"loss/{name}_lags_te_loss.txt",lags_te_loss_arr)
        np.savetxt(f"loss/{name}_lags_tr_loss.txt",lags_tr_loss_arr)
        np.savetxt(f"loss/{name}_lags_epochs.txt",loop_lags_epochs)

def grid_learning(device,wrk_dir,data_gen = False):
    """
    This method collates together methods to train neural networks utilizing a
    grid based learning strategy. This has less functionality than the active
    learning method as it is largely a method of comparison between the two
    strategies rather than a fully implemented method.

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
    print("Training using grid")
    rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
    rmf = unpack_rmf(rmf_name)
    egrid = rmf.e_min #energy grid used to evaluate the xspec model
    lags_egrid = np.logspace(np.log10(0.5),np.log10(11),num=26)
    
    pars_list = [1,2,3,4,13]
    negatives = [3]
    logged = [1,2,3,4]
    
    grid_sizes = [5,6,7,8,9,10]
    grid_names = []
    for size in grid_sizes:
        grid_names.append(f"grid_{size}")
    
    if data_gen == True:
        for fname,size in zip(grid_names,grid_sizes):
            generator.grid_data_gen(size, fname, egrid, lags_egrid)
    
    batch_size = 1024
    num_workers = 4
    
    modes = ["flux", "lags"]
    
    for (size,fname) in zip(grid_sizes,grid_names):
        print(f"Starting {size} x {size} grid loop")
        for mode in modes:
            print(f"Training on {mode}")
            if mode == "flux":
                train = train_flux
                test = test_flux
                dataType = PCADataset
            elif mode == "lags":
                train = train_lags
                test = test_lags
                dataType = PCALagsDataset
            training_data = dataType(f"data/locations/loc_{fname}_{mode}.csv",
                                     pars_list,negatives,logged,
                                     PCA_loc=f"scalers/PCA_{fname}_{mode}.bin",
                                     comp_loc=f"scalers/comp_{fname}_{mode}.bin")
            testing_data = dataType(f"data/locations/loc_{fname}_{mode}_test.csv",
                                    pars_list,negatives,logged,
                                    scale_bool=False,
                                    PCA_loc=f"scalers/PCA_{fname}_{mode}.bin",
                                    comp_loc=f"scalers/comp_{fname}_{mode}.bin")
            train_dataloader = DataLoader(training_data,batch_size=batch_size,
                                          num_workers = num_workers, shuffle=True)
            
            test_dataloader = DataLoader(testing_data,batch_size=batch_size,
                                          num_workers = num_workers, shuffle=True)
            print("Dataloaders created")
            
            loss_fn = PCALoss(training_data.pca.explained_variance_ratio_,device)
            
            if mode == "flux":
                model = network.PCAFluxNetwork(5,training_data.data.shape[1])
            elif mode == "lags":
                model = network.PCALagsNetwork(5,training_data.data.shape[1])
            model.to(device)
            optimizer = Adam(model.parameters(),lr = 0.001)
            
            grid_training_loop(model, optimizer, train, test, 
                                   train_dataloader, test_dataloader,
                                   loss_fn, device,
                                   size, mode)
def main():
    """
    Calls other methods when the main program is run. Makes sure all environmental
    variables for rtdist and CUDA/pytorhc are set.

    Returns
    -------
    None.

    """
    
    wrk_dir = os.getcwd()
    
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
    
    print("Environmental variables successfully set")
    print("Cuda is available:",torch.cuda.is_available())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    active_learning(device,wrk_dir,"PCA_active")
    grid_learning(device,wrk_dir,data_gen=True)
    
if __name__ == "__main__":
    main()
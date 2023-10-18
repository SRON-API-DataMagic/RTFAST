"""
This is the main program that trains the neural network.
"""
import numpy as np
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sherpa.astro.ui import unpack_rmf
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

from dataStructures import FluxData, LagsData

from processing import renameData
import generator
import network

from training import train_flux, train_lags, test_flux, test_lags
from training import active_training_loop, grid_training_loop
from training import FluxLoss, LagLoss
from training import QBDC
from generator import intialize_dataset

def active_learning(wrk_dir, device = "cpu"):
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
    
    active_loops = 40
    range_BH = np.asarray(generator.lhs_BH())
    range_AGN = np.asarray(generator.lhs_AGN())
    range_all = np.asarray(generator.lhs_all())
    
    theta_bh = generator.lhs_generation(10000, range_BH)
    theta_agn = generator.lhs_generation(10000, range_AGN)

    #generate physical models of test set
    theta_bh = generator.pars_conversion_full(theta_bh,0)
    theta_agn = generator.pars_conversion_full(theta_agn,0)
    
    with Parallel(n_jobs=10,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        flux_bh = []
        for pars in theta_bh:
            flux_bh.append(generator.rtdist_erg_flux(pars, egrid))
        flux_bh = np.asarray(flux_bh)
        flux_agn = parallel(delayed(generator.rtdist_erg_flux)(pars, egrid)
                                        for pars in theta_agn)
        flux_agn = np.asarray(flux_agn)
    
    plt.hist(flux_bh, bins=100)
    plt.axvline(2.4e-6, ls = "--")
    plt.axvline(1e-15, ls = "--")
    plt.title("Black hole flux distributions")
    plt.xscale("log")
    plt.savefig("bh_dists.png")
    plt.close()
    
    plt.hist(flux_agn, bins=100)
    plt.axvline(2.4e-6,ls = "--",c="b")
    plt.axvline(1e-15,ls = "--",c="g")
    plt.title("AGN flux distributions")
    plt.xscale("log")
    plt.savefig("agn_dists.png")
    plt.close()
    
    quit()
    
    labels = ["height","a","inc","rin","rout","z","Gamma","distance","Afe","logNe","kte",
              "nH","boost","mass","honr","b1","b2","phiAB","g","Anorm"]
    labels = ["a","inc","rin","rout","mass"]
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [0,3]
    logged = [0,2,3,4,7,8,10,11,12,13,19]
    
    pars_list = [1,2,3,4,13]
    negatives = [2]
    logged = [1,2,3,4]
    
    #pre generate Latin Hypercube samples.
    theta_bh = generator.lhs_generation(5e7, range_BH)
    theta_agn = generator.lhs_generation(5e7, range_AGN)
    
    theta_lhs = np.concat(theta_bh,theta_agn,axis=0)
    
    #if the first time running this code or you want to refresh the dataset, 
    #make this true
    first = True
    
    flux_name = "active_locs_flux.csv"
    flux_test_name = "active_test_locs_flux.csv"
    flux_scaler_name = "active_scaler_flux.bin"
    lags_name = "active_locs_lags.csv"
    lags_test_name = "active_test_locs_lags.csv"
    lags_scaler_name = "active_scaler_lags.bin"
    
    num_pars = range_BH.shape[0]
    
    flux_model = network.HeavyFluxNetwork(num_pars,len(egrid))
    flux_model.to(device)
    lags_model = network.HeavyLagsNetwork(num_pars,len(lags_egrid)-1)
    lags_model.to(device)
    
    best_flux_model = network.HeavyFluxNetwork(num_pars,len(egrid))
    best_flux_model.to(device)
    best_lags_model = network.HeavyLagsNetwork(num_pars,len(lags_egrid)-1)
    best_lags_model.to(device)
    
    optimizer_flux = Adam(flux_model.parameters(),lr = 5e-4)
    optimizer_lags = Adam(lags_model.parameters(),lr = 5e-4)
    #scheduler_flux = ReduceLROnPlateau(optimizer_flux,factor=0.5,patience=30)
    #scheduler_lags = ReduceLROnPlateau(optimizer_lags,factor=0.5,patience=30)
    scaler = MinMaxScaler()
    
    if first == True: 
        intialize_dataset(range_all, egrid, lags_egrid, flux_name, lags_name)
        
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
        flux_dataloader = FluxData(f"data/locations/{flux_name}", 
                                       scaler, flux_scaler_name,  
                                       pars_list=pars_list,
                                       negatives=negatives,logged=logged,
                                       scaling=True)
        lags_dataloader = LagsData(f"data/locations/{lags_name}", 
                                       scaler, lags_scaler_name,
                                       pars_list=pars_list,
                                       negatives=negatives,logged=logged,
                                       scaling=True)
        
        loss_fn_flux = FluxLoss(f"{flux_scaler_name}", device)
        loss_fn_lags = LagLoss(f"{lags_scaler_name}", device)
    else:
        name_num = 19
        active_loop_num = name_num+1
        flux_model.load_state_dict(torch.load(f"models/{name_num}_flux_model.pth"))
        lags_model.load_state_dict(torch.load(f"models/{name_num}_lags_model.pth"))
        
        optimizer_flux.load_state_dict(torch.load(f"models/{name_num}_flux_optimizer.pth"))
        optimizer_lags.load_state_dict(torch.load(f"models/{name_num}_lags_optimizer.pth"))
        
        loss_fn_flux = FluxLoss(f"{flux_scaler_name}", device)
        loss_fn_lags = LagLoss(f"{lags_scaler_name}", device)
        
        renameData(pd.read_csv(f"data/locations/loc_flux_{name_num}.csv"),"data/locations/",flux_name)
        renameData(pd.read_csv(f"data/locations/loc_lags_{name_num}.csv"),"data/locations/",lags_name)
        
        flux_tr_loss_arr = np.loadtxt(f"loss/{name_num}_flux_tr_loss.txt").tolist()
        flux_te_loss_arr = np.loadtxt(f"loss/{name_num}_flux_te_loss.txt").tolist()
        lags_tr_loss_arr = np.loadtxt(f"loss/{name_num}_lags_tr_loss.txt").tolist()
        lags_te_loss_arr = np.loadtxt(f"loss/{name_num}_lags_te_loss.txt").tolist()
        
        last_sig_flux_tr = np.min(flux_tr_loss_arr)
        last_sig_flux_te = np.min(flux_te_loss_arr)
        last_sig_lags_tr = np.min(lags_tr_loss_arr)
        last_sig_lags_te = np.min(lags_te_loss_arr)
        
        loop_flux_epochs = np.loadtxt(f"loss/{name_num}_flux_epochs.txt").tolist()
        loop_lags_epochs = np.loadtxt(f"loss/{name_num}_flux_epochs.txt").tolist()
        
    batch_size = 1024
    num_workers = 4
    
    lhs_idx = 0
    
    dec_mag = False
    
    print("Beginning training")
    with Parallel(n_jobs=10,verbose=5) as parallel:
        while active_loop_num <= active_loops:
            theta_lhs, lhs_idx = QBDC(flux_name, flux_test_name, lags_name, 
                                      lags_test_name, active_loop_num, 
                                      theta_lhs, lhs_idx, egrid, lags_egrid, 
                                      flux_model, lags_model, dec_mag, device, 
                                      labels, parallel)
    
            print("Setting up modeling")
            Xquery = FluxData(f"data/locations/{flux_name}", scaler, 
                              flux_scaler_name, pars_list=pars_list,
                              negatives=negatives,logged=logged)
            print("Query data set created")
            flux_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            print("Query data loader created")
            
            Xquery = LagsData(f"data/locations/{lags_name}", scaler, 
                              lags_scaler_name, pars_list=pars_list,
                              negatives=negatives,logged=logged)
            print("Query data set created")
            lags_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            print("Query data loader created")
            
            Xtest = FluxData(f"data/locations/{flux_test_name}", scaler, 
                              flux_scaler_name, pars_list=pars_list, 
                              negatives=negatives,logged=logged)
            print("Test data set created")
            flux_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                         num_workers = num_workers, shuffle=True)
            print("Test data loader created")
            
            Xtest = LagsData(f"data/locations/{lags_test_name}", scaler, 
                              lags_scaler_name, pars_list=pars_list, 
                              negatives=negatives,logged=logged)
            print("Test data set created")
            lags_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                         num_workers = num_workers, shuffle=True)
            print("Test data loader created")
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
                                                              train_lags, test_lags,
                                                              mode = "lags", dec_mag=dec_mag)
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
                                                              mode = "flux", dec_mag=dec_mag)
            
            
            #iterate loop number by 1
            active_loop_num += 1
            
        print("Completed training")
        print("Final best flux training loss:", last_sig_flux_tr)
        print("Final best flux testing loss:", last_sig_flux_te)
        torch.save(flux_model.state_dict(), "models/active_flux_final.pth")
        print("Saved PyTorch Model State to models/active_flux_final.pth")
        
        flux_tr_loss_arr = np.asarray(flux_tr_loss_arr)
        flux_te_loss_arr = np.asarray(flux_te_loss_arr)
        
        np.savetxt("loss/active_flux_te_loss.txt",flux_te_loss_arr)
        np.savetxt("loss/active_flux_tr_loss.txt",flux_tr_loss_arr)
        np.savetxt("loss/active_flux_epochs.txt",loop_flux_epochs)
        
        print("Final best lags training loss:", last_sig_lags_tr)
        print("Final best lags testing loss:", last_sig_lags_te)
        torch.save(lags_model.state_dict(), "models/active_lags_final.pth")
        print("Saved PyTorch Model State to models/active_lags_final.pth")
        
        lags_tr_loss_arr = np.asarray(lags_tr_loss_arr)
        lags_te_loss_arr = np.asarray(lags_te_loss_arr)
        
        np.savetxt("loss/active_lags_te_loss.txt",lags_te_loss_arr)
        np.savetxt("loss/active_lags_tr_loss.txt",lags_tr_loss_arr)
        np.savetxt("loss/active_lags_epochs.txt",loop_lags_epochs)

def grid_learning(wrk_dir,device):
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
    
    locations = "data/locations/"
    
    pars_list = [1,2,3,4,7]
    negatives = [2]
    logged = [1,2,3,4]
    
    grid_sizes = [5,6,7,8,9,10]
    grid_names = []
    for size in grid_sizes:
        grid_names.append(f"grid_{size}")
    print(grid_names)
        
    batch_size = 1024
    num_workers = 4
    
    modes = ["flux", "lags"]
    
    for (size,fname) in zip(grid_sizes,grid_names):
        print(f"Starting {size} x {size} grid loop")
        for mode in modes:
            print(f"Training on {mode}")
            
            scaler = MinMaxScaler()
            #create initial dataset object to create scaler (and then delete object)
            
            if mode == "flux":
                train = train_flux
                test = test_flux
                dataType = FluxData
                model = network.HeavyFluxNetwork(5,len(egrid))
            elif mode == "lags":
                train = train_lags
                test = test_lags
                dataType = LagsData
                model = network.HeavyLagsNetwork(5,len(lags_egrid)-1)
            else:
                print("Invalid mode, defaulting to flux modelling")
                train = train_flux
                test = test_flux
                dataType = FluxData
                model = network.MagFluxNetwork(5,len(egrid))
            
            model.to(device)
            optimizer = Adam(model.parameters(),lr = 0.001)
            
            training_data = dataType(locations+f"loc_{fname}_{mode}.csv", scaler, 
                             scaler_name=f"{fname}_{mode}_scaler.bin", 
                             pars_list=pars_list,
                             negatives=negatives, logged=logged, scaling=True)
        
            train_dataloader = DataLoader(training_data,batch_size=batch_size,
                                          num_workers = num_workers, shuffle=True)
            
            testing_data = dataType(locations+f"loc_{fname}_{mode}_test.csv", scaler, 
                                 scaler_name=f"{fname}_{mode}_scaler.bin", 
                                 pars_list=pars_list,
                                 negatives=negatives,logged=logged)
            
            test_dataloader = DataLoader(testing_data,batch_size=batch_size,
                                          num_workers = num_workers, shuffle=True)
            
            if mode == "flux":
                loss_fn = FluxLoss(f"{fname}_{mode}_scaler.bin", 
                                         device)
            elif mode == "lags":
                loss_fn = LagLoss(f"{fname}_{mode}_scaler.bin", 
                                         device)
            
            print("Dataloaders created")
            
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
    torch.set_default_dtype(torch.double)
    
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
    
    active_learning(wrk_dir,device)
    grid_learning(wrk_dir,device)
    

if __name__ == "__main__":
    main()
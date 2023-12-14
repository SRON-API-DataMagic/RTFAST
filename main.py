"""
This is the main program that trains the neural network.
"""
import numpy as np
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sherpa.astro.ui import unpack_rmf

import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

def active_learning(device, wrk_dir, name, world_size=1, parallelism=False):
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
    range_AGN = np.asarray(generator.lhc_trimmed_gen())
    
    labels = ["height","a","inc","rin","rout","z","Gamma","distance","Afe","logNe","kte",
              "nH","boost","mass","honr","b1","b2","phiAB","g","Anorm"]
    labels = ["a","inc","rin","distance","mass"]
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,19]
    
    pars_list = [1,2,3,4,13]
    negatives = [2]
    logged = [1,2,3,4]
    
    lhc_idx = 0
    theta_lhc = generator.lhc_generation(int(1e6), range_AGN, limited = True)
    #if the first time running this code or you want to refresh the dataset, 
    #make this true
    first = True
    
    flux_name = f"{name}_locs_flux.csv"
    flux_test_name = f"{name}_test_locs_flux.csv"
    flux_scaler_name = f"{name}_scaler_flux.bin"
    lags_name = f"{name}_locs_lags.csv"
    lags_test_name = f"{name}_test_locs_lags.csv"
    lags_scaler_name = f"{name}_scaler_lags.bin"
    
    num_pars = range_AGN.shape[0]
    
    if parallelism == False:
        flux_model = network.HeavyFluxNetwork(num_pars,len(egrid))
        flux_model.to(device)
        lags_model = network.HeavyLagsNetwork(num_pars,len(lags_egrid)-1)
        lags_model.to(device)
        
        best_flux_model = network.HeavyFluxNetwork(num_pars,len(egrid))
        best_flux_model.to(device)
        best_lags_model = network.HeavyLagsNetwork(num_pars,len(lags_egrid)-1)
        best_lags_model.to(device)
    else:
        flux_model = DDP(network.HeavyFluxNetwork, device_ids=["dead"])
    
    optimizer_flux = Adam(flux_model.parameters(),lr = 5e-4)
    optimizer_lags = Adam(lags_model.parameters(),lr = 5e-4)
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    def read_data(csv):
        locations = csv.iloc[:,-1]
        data = []
        for location in tqdm.tqdm(locations):
            datum = np.loadtxt(location).reshape(1, -1)
            data.append(datum)
        data = np.asarray(data)
        return data
    
    if first == True: 
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
        
        flux_pars = pd.read_csv(f"data/locations/{flux_name}")
        flux_data = read_data(flux_pars)
        
        lags_pars = pd.read_csv(f"data/locations/{lags_name}")
        lags_data = read_data(lags_pars)
        
        #create initial dataset object to create scaler
        flux_dataloader = FluxData(flux_pars, flux_data,
                                       scaler, flux_scaler_name,  
                                       pars_list=pars_list,
                                       negatives=negatives,logged=logged,
                                       scaling=True)
        lags_dataloader = LagsData(lags_pars, lags_data, 
                                       scaler, lags_scaler_name,
                                       pars_list=pars_list,
                                       negatives=negatives,logged=logged,
                                       scaling=True)
        
        loss_fn_flux = FluxLoss(f"{flux_scaler_name}", device)
        loss_fn_lags = LagLoss(f"{lags_scaler_name}", device)
    else:
        name_num = 30
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
        loop_lags_epochs = np.loadtxt(f"loss/{name_num}_lags_epochs.txt").tolist()
        
    batch_size = 1024
    num_workers = 4
    
    print("Beginning training")
    with Parallel(n_jobs=20,verbose=3) as parallel:
        while active_loop_num <= active_loops:
            theta_lhc, lhc_idx = QBDC(flux_name, flux_test_name, lags_name, 
                                      lags_test_name, active_loop_num, 
                                      theta_lhc, lhc_idx, egrid, lags_egrid, 
                                      flux_model, lags_model, device, 
                                      labels, parallel, parallelism)
            
            print("loading flux data")
            flux_pars = pd.read_csv(f"data/locations/{flux_name}")
            flux_data = read_data(flux_pars)
            
            print("loading flux test data")
            flux_test_pars = pd.read_csv(f"data/locations/{flux_test_name}")
            flux_test_data = read_data(flux_test_pars)
            
            print("loading lags data")
            lags_pars = pd.read_csv(f"data/locations/{lags_name}")
            lags_data = read_data(lags_pars)
            
            print("loading lags test data")
            lags_test_pars = pd.read_csv(f"data/locations/{lags_test_name}")
            lags_test_data = read_data(lags_test_pars)
            
            print("Setting up modeling")
            Xquery = FluxData(flux_pars, flux_data, scaler, 
                              flux_scaler_name, pars_list=pars_list,
                              negatives=negatives,logged=logged)
            print("Query data set created")
            flux_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            print("Query data loader created")
            
            Xquery = LagsData(lags_pars, lags_data, scaler, 
                              lags_scaler_name, pars_list=pars_list,
                              negatives=negatives,logged=logged)
            print("Query data set created")
            lags_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            print("Query data loader created")
            
            Xtest = FluxData(flux_test_pars, flux_test_data, scaler, 
                              flux_scaler_name, pars_list=pars_list, 
                              negatives=negatives,logged=logged)
            print("Test data set created")
            flux_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                         num_workers = num_workers, shuffle=True)
            print("Test data loader created")
            
            Xtest = LagsData(lags_test_pars, lags_test_data, scaler, 
                              lags_scaler_name, pars_list=pars_list, 
                              negatives=negatives,logged=logged)
            print("Test data set created")
            lags_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                         num_workers = num_workers, shuffle=True)
            print("Test data loader created")
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
                                                              train_lags, test_lags,
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
    if parallelism == True:
        destroy_process_group()

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
    
    locations = "data/locations/"
    
    pars_list = [1,2,3,4,13]
    negatives = [2]
    logged = [1,2,3,4]
    
    grid_sizes = [5,6,7,8,9,10]
    grid_names = []
    for size in grid_sizes:
        grid_names.append(f"grid_{size}")
    print(grid_names)
    
    if data_gen == True:
        for fname,size in zip(grid_names,grid_sizes):
            generator.grid_data_gen(size, fname, egrid, lags_egrid)
    
    batch_size = 1024
    num_workers = 4
    
    modes = ["flux", "lags"]
    
    def read_data(csv):
        locations = csv.iloc[:,-1]
        data = []
        for location in locations:
            datum = np.loadtxt(location).reshape(1, -1)
            data.append(datum)
        data = np.asarray(data)
        return data
    
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
            
            train_pars = pd.read_csv(f"data/locations/loc_{fname}_{mode}.csv")
            train_data = read_data(train_pars)
            
            test_pars = pd.read_csv(f"data/locations/loc_{fname}_{mode}_test.csv")
            test_data = read_data(test_pars)
            
            training_data = dataType(train_pars, train_data, scaler, 
                             scaler_name=f"{fname}_{mode}_scaler.bin", 
                             pars_list=pars_list,
                             negatives=negatives, logged=logged, scaling=True)
        
            train_dataloader = DataLoader(training_data,batch_size=batch_size,
                                          num_workers = num_workers, shuffle=True)
            
            testing_data = dataType(test_pars, test_data, scaler, 
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

def fixed_data_varied_training(device,wrk_dir):
    """
    This method is used for the exploration of the effect of the use of
    different scaling and learning rate schedulers on training of the emulator.
    It is intended to only change these two aspects and uses data previously
    generated by active learning cycles. This reduces the total time to
    train the network as there is no longer a need to generate new data but
    does accordingly miss out on the benefits of active learning over time.

    Returns
    -------
    None.

    """
    print("Training using grid")
    rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
    rmf = unpack_rmf(rmf_name)
    egrid = rmf.e_min #energy grid used to evaluate the xspec model
    lags_egrid = np.logspace(np.log10(0.5),np.log10(11),num=26)
    
    batch_size = 1024
    num_workers = 4
    
    locations = "data/locations/"
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,19]
    
    num_pars = len(pars_list)
    
    flux_name = "active_locs_flux.csv"
    flux_test_name = "active_test_locs_flux.csv"
    lags_name = "active_locs_lags.csv"
    lags_test_name = "active_test_locs_lags.csv"
    
    training_epochs = 50
    
    optimizers = ["adam","sgd"]
    schedulers = ["fixed","cyclic"]
    scalers = ["minmax","std"]
    
    for scaler_typ in scalers:
        print(f"Attempting scaler type {scaler_typ}")
        #select scalers
        if scaler_typ == "minmax":
            scaler_flux = MinMaxScaler()
            scaler_lags = MinMaxScaler()
        elif scaler_typ == "std":
            scaler_flux = StandardScaler()
            scaler_lags = StandardScaler()
        else:
            print(f"{scaler_typ} is not a valid scaler, skipping...")
            continue
        
        flux_scaler_name = f"{scaler_typ}_scaler_flux.bin"
        lags_scaler_name = f"{scaler_typ}_scaler_lags.bin"
        
        Xquery = FluxData(f"data/locations/{flux_name}", scaler_flux, 
                          flux_scaler_name, pars_list=pars_list,
                          negatives=negatives,logged=logged, scaling=True,
                          end=30000)
        print("Flux training data set created")
        flux_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                      num_workers = num_workers, shuffle=True)
        print("Flux data loader created")
        
        Xquery = LagsData(f"data/locations/{lags_name}", scaler_lags, 
                          lags_scaler_name, pars_list=pars_list,
                          negatives=negatives,logged=logged, scaling=True,
                          end=30000)
        print("Lags training data set created")
        lags_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                      num_workers = num_workers, shuffle=True)
        print("Lags data loader created")
        
        Xtest = FluxData(f"data/locations/{flux_test_name}", scaler_flux, 
                          flux_scaler_name, pars_list=pars_list, 
                          negatives=negatives,logged=logged)
        print("Flux test data set created")
        flux_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                     num_workers = num_workers, shuffle=True)
        print("Flux test data loader created")
        
        Xtest = LagsData(f"data/locations/{lags_test_name}", scaler_lags, 
                          lags_scaler_name, pars_list=pars_list, 
                          negatives=negatives,logged=logged)
        print("Lags test data set created")
        lags_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                     num_workers = num_workers, shuffle=True)
        print("Lags test data loader created")
        
        loss_fn_flux = FluxLoss(flux_scaler_name, device)
        loss_fn_lags = LagLoss(lags_scaler_name, device)
        
        for scheduler_typ in schedulers:
            for optimizer_typ in optimizers:
                #instantiate models
                flux_model = network.HeavyFluxNetwork(num_pars,len(egrid))
                flux_model.to(device)
                lags_model = network.HeavyLagsNetwork(num_pars,len(lags_egrid)-1)
                lags_model.to(device)
                
                base_lr = 0.00001
                max_lr = 0.001
                step_size_up = len(flux_test_dataloader)*5
                
                #select optimizers
                if optimizer_typ == "adam":
                    optimizer_flux = Adam(flux_model.parameters(),lr = max_lr)
                    optimizer_lags = Adam(lags_model.parameters(),lr = max_lr)
                elif optimizer_typ == "sgd":
                    optimizer_flux = SGD(flux_model.parameters(),lr = max_lr)
                    optimizer_lags = SGD(lags_model.parameters(),lr = max_lr)
                else:
                    print("No valid optimizers, skipping...")
                    continue
                
                #select schedulers - this is done per model as this effects 
                #individual optimizers
                if scheduler_typ == "fixed":
                    scheduler_flux = None
                    scheduler_lags = None
                elif scheduler_typ == "cyclic":
                    scheduler_flux = CyclicLR(optimizer_flux,
                                              base_lr,max_lr,
                                              step_size_up)
                    scheduler_lags = CyclicLR(optimizer_lags,
                                              base_lr,max_lr,
                                              step_size_up)
                else:
                    print("No valid schedulers, skipping...")
                    continue
                
                grid_training_loop(flux_model, optimizer_flux, train_flux, 
                                   test_flux, flux_dataloader, 
                                   flux_test_dataloader,
                                   loss_fn_flux, device,
                                   f"{scaler_typ}_{optimizer_typ}_{scheduler_typ}", 
                                   "flux", training_epochs)
                
                grid_training_loop(lags_model, optimizer_lags, train_lags, 
                                   test_lags, lags_dataloader, 
                                   lags_test_dataloader,
                                   loss_fn_lags, device,
                                   f"{scaler_typ}_{optimizer_typ}_{scheduler_typ}", 
                                   "lags", training_epochs)

def ddp_setup(rank: int, world_size: int):
  """
  Sets up distributed GPU processing.
  
  Parameters
  ----------
     rank: int
         Unique identifier of each process
     world_size: int
         Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

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
    
    parallelism = False
    print("Cuda is available:",torch.cuda.is_available())
    
    if parallelism == False:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        world_size = torch.cuda.device_count()
        mp.spawn(active_learning, args=(wrk_dir,world_size,parallelism), 
                 nprocs=world_size)

    active_learning(device,wrk_dir,"short_active")
    grid_learning(device,wrk_dir,data_gen=True)
    #fixed_data_varied_training(device,wrk_dir)
    

if __name__ == "__main__":
    main()
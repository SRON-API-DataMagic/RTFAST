"""
This is the main program that trains the neural network.
"""
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt

from sherpa.astro.ui import unpack_rmf
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import scipy.stats

from dataStructures import FluxDecData, LagsDecData

from processing import nanChecker, saveData, renameData
import generator
import network

from training import train_flux, train_lags, test_flux, test_lags, barredMSELoss
from training import active_training_loop, grid_training_loop, lagLoss
from training import magDecFluxLoss, magDecLagsLoss
from plotting import distributions
from generator import active_learning_generation, pars_conversion, pars_conversion_full
#from generator import grid_data_gen

def queryByDropout(wrk_dir, device = "cpu"):
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
    range_all = np.asarray(generator.lhs_trimmed_gen())
    
    labels = ["height","a","inc","rin","rout","z","Gamma","distance","Afe","logNe","kte",
              "nH","boost","mass","honr","b1","b2","phiAB","g","Anorm"]
    labels = ["a","mass","inc","rin","rout"]
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [0,3]
    logged = [0,2,3,4,7,8,10,11,12,13,19]
    
    pars_list = [1,13,2,3,4]
    negatives = [3]
    logged = [1,2,3,4]
    
    #pre generate Latin Hypercube samples.
    sampler = scipy.stats.qmc.LatinHypercube(d=len(range_all))
    sample = sampler.random(n=1000000)
    theta_lhs = scipy.stats.qmc.scale(sample, range_all[:,0], range_all[:,1])
    
    #if the first time running this code or you want to refresh the dataset, 
    #make this true
    first = True
    
    flux_name = "active_locs_flux.csv"
    flux_test_name = "active_test_locs_flux.csv"
    flux_scaler_name = "active_scaler_flux.bin"
    lags_name = "active_locs_lags.csv"
    lags_test_name = "active_test_locs_lags.csv"
    lags_scaler_name = "active_scaler_lags.bin"
    
    num_pars = range_all.shape[0]
    
    flux_model = network.MagFluxNetwork(num_pars,len(egrid))
    flux_model.to(device)
    lags_model = network.MagLagsNetwork(num_pars,len(lags_egrid)-1)
    lags_model.to(device)
    
    best_flux_model = network.MagFluxNetwork(num_pars,len(egrid))
    best_flux_model.to(device)
    best_lags_model = network.MagLagsNetwork(num_pars,len(lags_egrid)-1)
    best_lags_model.to(device)
    
    optimizer_flux = Adam(flux_model.parameters(),lr = 5e-4)
    optimizer_lags = Adam(lags_model.parameters(),lr = 5e-4)
    #scheduler_flux = ReduceLROnPlateau(optimizer_flux,factor=0.5,patience=30)
    #scheduler_lags = ReduceLROnPlateau(optimizer_lags,factor=0.5,patience=30)
    scaler = MinMaxScaler()
    start_num = 0
    
    if first == True: 
        print("Generating first time dataset")
        init_data_size = 5000
        #generating a random set of parameters and corresponding data
        theta_init = np.random.uniform(range_all[:,0],range_all[:,1],
                                       size = (init_data_size,range_all.shape[0]))
        pars_init_flux = generator.pars_conversion(theta_init,0)
        pars_init_lags = generator.pars_conversion(theta_init,6)
        print("Parallelized model generation")
        flux_data_init =  Parallel(n_jobs=10,verbose=5)(delayed(generator.rtdist_flux)(pars, egrid)
                                        for pars in pars_init_flux)
        lags_data_init =  Parallel(n_jobs=10,verbose=5)(delayed(generator.rtdist_lags)(pars, lags_egrid)
                                        for pars in pars_init_lags)
        flux_data_init = np.array(flux_data_init)
        lags_data_init = np.array(lags_data_init)
        
        #check for and delete parameter sets producing NaN results for flux
        index = nanChecker(flux_data_init, theta_init)
        flux_data_init = np.delete(flux_data_init, index, axis=0)
        lags_data_init = np.delete(lags_data_init, index, axis=0)
        pars_init_flux = np.delete(pars_init_flux, index, axis=0)
        pars_init_lags = np.delete(pars_init_lags, index, axis=0)
        
        #check for and delete parameter sets producing NaN results for time lags
        index = nanChecker(lags_data_init, theta_init)
        flux_data_init = np.delete(flux_data_init, index, axis=0)
        lags_data_init = np.delete(lags_data_init, index, axis=0)
        pars_init_flux = np.delete(pars_init_flux, index, axis=0)
        pars_init_lags = np.delete(pars_init_lags, index, axis=0)
        
        #save data for the first time in text files
        saveData(flux_data_init, pars_init_flux, "data/locations/", flux_name)
        saveData(lags_data_init, pars_init_lags, "data/locations/", lags_name, 
                 lags = True)
        
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
        flux_dataloader = FluxDecData(f"data/locations/{flux_name}", 
                                       scaler, flux_scaler_name,  
                                       pars_list=pars_list,
                                       negatives=negatives,logged=logged,
                                       scaling=True)
        lags_dataloader = LagsDecData(f"data/locations/{lags_name}", 
                                       scaler, lags_scaler_name,
                                       pars_list=pars_list,
                                       negatives=negatives,logged=logged,
                                       scaling=True)
        
        loss_fn_flux = magDecFluxLoss(f"dec_{flux_scaler_name}",
                                      f"mag_{flux_scaler_name}", device)
        loss_fn_lags = magDecLagsLoss(f"dec_{lags_scaler_name}",
                                      f"mag_{lags_scaler_name}", device)
        
    else: #load previously generated data as initial data and parameter set
        start_num = 30
        active_loop_num = start_num + 1
        print("Loading previous data")
        renameData(pd.read_csv(f"data/locations/loc_{start_num}.csv"), 
                   "data/locations/", "active_locs.csv")
        
        with open(f"loss/{start_num}_tr_loss.txt","r") as f1:
            tr_loss_arr = np.loadtxt(f1)
        f1.close()
        
        with open(f"loss/{start_num}_te_loss.txt","r") as f1:
            te_loss_arr = np.loadtxt(f1)
        f1.close()
        
        with open(f"loss/{start_num}_epochs.txt","r") as f1:
            loop_epochs = np.loadtxt(f1)
        f1.close()
        
        last_sig_best_tr = tr_loss_arr.min()
        last_sig_best_te = te_loss_arr.min()
        print("Best training loss:",last_sig_best_tr)
        print("Best testing loss:",last_sig_best_te)
        
        tr_loss_arr = tr_loss_arr.tolist()
        te_loss_arr = te_loss_arr.tolist()
        loop_epochs = loop_epochs.tolist()
        
        flux_model.load_state_dict(torch.load(f"models/{start_num}_model.pth"))
        optimizer_flux.load_state_dict(torch.load(f"models/{start_num}_optimizer.pth"))
        #use different loss function
        loss_fn_flux = barredMSELoss("active_scaler.bin",device)
        
    batch_size = 1024
    num_workers = 4
    
    lhs_idx = 0
    
    dec_mag = True
    
    print("Beginning training")
    with Parallel(n_jobs=10,verbose=5) as parallel:
        while active_loop_num <= active_loops:
            data_size = len(pd.read_csv(f"data/locations/{flux_name}"))
            multiplier = ceil(data_size/100000)
            n_samples = 5000*multiplier
            n_samples_large = 10000*multiplier # number of parameter sets to draw 
            divider = 100*multiplier
            n_samples_small = int(n_samples_large/divider)
            print(f"I am in active learning loop {active_loop_num}")
            # randomly generate points in parameter space
            print("Generating random samples of theta")
            theta_query_large = theta_lhs[lhs_idx : lhs_idx+n_samples_large]
            
            print("computing neural network predictions with dropout for each theta")
            # compute 100 neural network predictions with dropout
            sample_dropout = 100
            pred_query_flux = np.zeros((sample_dropout,n_samples_small,len(egrid)))
            pred_query_lags = np.zeros((sample_dropout,n_samples_small,len(lags_egrid)-1))
            pred_query_inds = np.zeros((sample_dropout,n_samples_small,len(lags_egrid)-1))
            flux_model.train()
            lags_model.train()
            query_samples = []
            
            for j in tqdm(range(divider),desc="Sample dropout loops"):
                theta_query_small = theta_query_large[j*n_samples_small:(j+1)*n_samples_small]
                for i in range(sample_dropout):
                    if dec_mag == True:
                        mag_flux, dec_flux = flux_model(torch.DoubleTensor(theta_query_small).to(device))
                        mag_lags, dec_lags, ind = lags_model(torch.DoubleTensor(theta_query_small).to(device))
                        pred_query_flux[i] = mag_flux.detach().cpu().numpy() + dec_flux.detach().cpu().numpy()
                        pred_query_lags[i] = mag_lags.detach().cpu().numpy() + dec_lags.detach().cpu().numpy()
                        pred_query_inds[i] = ind.detach().cpu().numpy()
                    else:
                        pred_flux = flux_model(torch.DoubleTensor(theta_query_small).to(device))
                        pred_lags, ind = lags_model(torch.DoubleTensor(theta_query_small).to(device))
                        pred_query_flux[i] = pred_flux.detach().cpu().numpy()
                        pred_query_lags[i] = pred_lags.detach().cpu().numpy()
                        pred_query_inds[i] = ind.detach().cpu().numpy()
                # find uncertainty (as measured by relative variance)
                dvar_flux = np.var(pred_query_flux,axis=0)
                mean_var_flux = np.mean(dvar_flux, axis=1)
                # find uncertainty (as measured by relative variance)
                dvar_lags = np.var(pred_query_lags,axis=0)
                mean_var_lags = np.mean(dvar_lags, axis=1)
                # find uncertainty (as measured by relative variance)
                dvar_inds = np.var(pred_query_inds,axis=0)
                mean_var_inds = np.mean(dvar_inds, axis=1)
                #sum the two
                mean_var_query = mean_var_flux+0.5*(mean_var_lags+mean_var_inds)
                # add to uncertainties per theta to list
                query_samples.append(mean_var_query.tolist())
            
            #Performing manual memory cleanup
            print("Successfully finished generating thetas")
            
            print("Finding top uncertain thetas")
            # sort these thetas from smallest uncertainty to largest and save values
            query_samples = np.asarray(query_samples).flatten()
            np.savetxt(f"dists/loop_{active_loop_num}_variances.txt",query_samples)
            query_idx = np.argsort(query_samples)[::-1]
            
            #Plot distribution of variances
            plt.hist(query_samples,bins=100)
            plt.savefig(f"dists/loop_{active_loop_num}_variances.png")
            plt.close()
            
            print("Top sample mean variance",query_samples[query_idx[0]])
            print("Bottom sample mean variance",query_samples[query_idx[-1]])
            print("Range of mean variance",np.ptp(query_samples))
            
            print("Generating data for these samples")
            # get out the top `nsamples` values of theta_query
            theta_query = theta_query_large[query_idx[:n_samples]]
            
            distributions(theta_query, labels, f"dists/loop_{active_loop_num}_")
            
            active_learning_generation(theta_query, egrid, lags_egrid, parallel, 
                                           flux_name, flux_test_name, lags_name,
                                           lags_test_name, pars_conversion)
            
            # add rejected parameter sets back to original array for potential 
            # future use:
            theta_lhs = np.vstack([theta_lhs, theta_query_large[query_idx[n_samples:]]])
            
            # increment the index for reading parameters from theta_lhs
            lhs_idx += (n_samples_large)
    
            print("Setting up modeling")
            Xquery = FluxDecData(f"data/locations/{flux_name}", scaler, 
                              flux_scaler_name, pars_list=pars_list,
                              negatives=negatives,logged=logged)
            print("Query data set created")
            flux_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            print("Query data loader created")
            
            Xquery = LagsDecData(f"data/locations/{lags_name}", scaler, 
                              lags_scaler_name, pars_list=pars_list,
                              negatives=negatives,logged=logged)
            print("Query data set created")
            lags_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            print("Query data loader created")
            
            Xtest = FluxDecData(f"data/locations/{flux_test_name}", scaler, 
                              flux_scaler_name, pars_list=pars_list, 
                              negatives=negatives,logged=logged)
            print("Test data set created")
            flux_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                         num_workers = num_workers, shuffle=True)
            print("Test data loader created")
            
            Xtest = LagsDecData(f"data/locations/{lags_test_name}", scaler, 
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
                                                              mode = "flux", dec_mag=dec_mag)
            
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

def grid(wrk_dir,device):
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
    
    pars_list = [1,13,2,3,4]
    negatives = [3]
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
                loss_fn = barredMSELoss(f"{fname}_{mode}_scaler.bin",device)
                dataType = FluxDecData
                model = network.HeavyFluxNetwork(5,len(egrid))
            elif mode == "lags":
                train = train_lags
                test = test_lags
                loss_fn = lagLoss(f"{fname}_{mode}_scaler.bin",device)
                dataType = LagsDecData
                model = network.HeavyLagsNetwork(5,len(lags_egrid)-1)
                
            else:
                print("Invalid mode, defaulting to flux modelling")
                train = train_flux
                test = test_flux
                loss_fn = barredMSELoss(f"{fname}_{mode}_scaler.bin",device)
                dataType = FluxDecData
                model = network.HeavyFluxNetwork(5,len(egrid))
            
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
    
    print(torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #grid(wrk_dir,device)
    queryByDropout(wrk_dir,device)
    

if __name__ == "__main__":
    main()
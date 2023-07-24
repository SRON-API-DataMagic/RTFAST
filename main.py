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

from dataStructures import FluxData, LagsData

from processing import nanChecker, saveData, renameData
import generator
import network

from training import train_flux, train_lags, test_flux, test_lags, barredMSELoss
from training import active_training_loop, grid_training_loop, lagLoss
from plotting import distributions

def queryByDropout(wrk_dir, device = None):
    print("Training using query by dropout committee")
    rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
    rmf = unpack_rmf(rmf_name)
    egrid = rmf.e_min #energy grid used to evaluate the xspec model
    lags_egrid = np.logspace(np.log10(0.5),np.log10(11),num=26)
    
    active_loops = 30
    range_all = np.asarray(generator.lhs_trimmed_gen())
    
    labels = ["a","mass","inc","rin","rout"]
    
    #pre generate Latin Hypercube samples.
    sampler = scipy.stats.qmc.LatinHypercube(d=len(range_all))
    sample = sampler.random(n=1000000)
    theta_lhs = scipy.stats.qmc.scale(sample, range_all[:,0], range_all[:,1])
    
    #if the first time running this code or you want to refresh the dataset, 
    #make this true
    first = True
    
    flux_model = network.SharpNetwork(5,len(egrid))
    flux_model.to(device)
    lags_model = network.LagsNetwork(5,len(lags_egrid)-1)
    lags_model.to(device)
    
    best_flux_model = network.SharpNetwork(5,len(egrid))
    best_flux_model.to(device)
    best_lags_model = network.LagsNetwork(5,len(lags_egrid)-1)
    best_lags_model.to(device)
    
    optimizer_flux = Adam(flux_model.parameters(),lr = 5e-4)
    optimizer_lags = Adam(lags_model.parameters(),lr = 5e-4)
    #scheduler = ReduceLROnPlateau(optimizer,factor=0.5,patience=30)
    scaler = MinMaxScaler()
    start_num = 0
    
    if first == True: 
        print("Generating first time dataset")
        init_data_size = 5000
        #generating a random set of parameters and corresponding data
        theta_init = np.random.uniform(range_all[:,0],range_all[:,1],
                                       size = (init_data_size,range_all.shape[0]))
        pars_init = generator.pars_conversion(theta_init)
        print("Parallelized model generation")
        flux_data_init =  Parallel(n_jobs=10,verbose=5)(delayed(generator.rtdist_flux)(pars, egrid)
                                        for pars in pars_init)
        lags_data_init =  Parallel(n_jobs=10,verbose=5)(delayed(generator.rtdist_lags)(pars, lags_egrid)
                                        for pars in pars_init)
        flux_data_init = np.array(flux_data_init)
        lags_data_init = np.array(lags_data_init)
        
        #check for and delete parameter sets producing NaN results for flux
        index = nanChecker(flux_data_init, theta_init)
        flux_data_init = np.delete(flux_data_init,index, axis=0)
        lags_data_init = np.delete(lags_data_init,index, axis=0)
        pars_init = np.delete(pars_init,index, axis=0)
        
        #check for and delete parameter sets producing NaN results for time lags
        index = nanChecker(lags_data_init, theta_init)
        flux_data_init = np.delete(flux_data_init,index, axis=0)
        lags_data_init = np.delete(lags_data_init,index, axis=0)
        pars_init = np.delete(pars_init,index, axis=0)
        
        pars_init = generator.pars_conversion(theta_init)
        #save data for the first time in text files
        saveData(flux_data_init, pars_init, 
                 "data/locations/","active_locs_flux.csv")
        saveData(lags_data_init, pars_init, 
                 "data/locations/","active_locs_lags.csv",
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
        flux_dataloader = FluxData("data/locations/active_locs_flux.csv", 
                                       scaler,"active_scaler_flux.bin",
                                       scaling=True)
        lags_dataloader = LagsData("data/locations/active_locs_lags.csv", 
                                       scaler,"active_scaler_lags.bin",
                                       scaling=True)
        
        loss_fn_flux = barredMSELoss("active_scaler_flux.bin",device)
        loss_fn_lags = lagLoss("active_scaler_lags.bin",device)
        
    else: #load previously generated data as initial data and parameter set
        start_num = 30
        active_loop_num = start_num + 1
        print("Loading previous data")
        renameData(f"data/locations/loc_{start_num}.csv", "data/locations/", "active_locs.csv")
        
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
    
    print("Beginning training")
    with Parallel(n_jobs=10,verbose=5) as parallel:
        while active_loop_num <= active_loops:
            data_size = len(pd.read_csv("data/locations/active_locs_flux.csv"))
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
            flux_model.train()
            lags_model.train()
            query_samples = []
            
            for j in tqdm(range(divider),desc="Sample dropout loops"):
                theta_query_small = theta_query_large[j*n_samples_small:(j+1)*n_samples_small]
                for i in range(sample_dropout):
                    pred_flux = flux_model(torch.DoubleTensor(theta_query_small).to(device))
                    pred_lags = lags_model(torch.DoubleTensor(theta_query_small).to(device))
                    pred_query_flux[i] = pred_flux.detach().cpu().numpy()
                    pred_query_lags[i] = pred_lags.detach().cpu().numpy()
                # find uncertainty (as measured by relative variance)
                dvar_flux = np.var(pred_query_flux,axis=0)
                mean_var_flux = np.mean(dvar_flux, axis=1)
                # find uncertainty (as measured by relative variance)
                dvar_lags = np.var(pred_query_lags,axis=0)
                mean_var_lags = np.mean(dvar_lags, axis=1)
                #sum the two
                mean_var_query = mean_var_lags+mean_var_flux
                # add to uncertainties per theta to list
                query_samples.append(mean_var_query.tolist())
            
            #Performing manual memory cleanup
            del pred_flux, pred_lags, pred_query_flux, pred_query_lags 
            del theta_query_small, mean_var_flux, mean_var_lags, mean_var_query
            del dvar_flux, dvar_lags
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
            
            # compute the physical model for these thetas
            theta_query_iterate = generator.pars_conversion(theta_query)
            print("Parallelized model generation")
            data_query =  parallel(delayed(generator.rtdist_flux)(pars, egrid)
                                            for pars in theta_query_iterate)
            data_query = np.asarray(data_query)
            lags_query =  parallel(delayed(generator.rtdist_lags)(pars, lags_egrid)
                                            for pars in theta_query_iterate)
            lags_query = np.asarray(lags_query)
            del theta_query_iterate
            
            #Remove any broken models
            #check for and delete parameter sets producing NaN results for flux
            index = nanChecker(data_query, theta_query)
            data_query = np.delete(data_query,index, axis=0)
            lags_query = np.delete(lags_query,index, axis=0)
            theta_query = np.delete(theta_query,index, axis=0)
            
            #check for and delete parameter sets producing NaN results for time lags
            index = nanChecker(lags_query, theta_query)
            data_query = np.delete(data_query, index, axis=0)
            lags_query = np.delete(lags_query, index, axis=0)
            theta_query = np.delete(theta_query, index, axis=0)
            
            # shuffle indices for neural network training
            idx_shuffle = np.arange(0, len(theta_query), dtype=int)
            np.random.shuffle(idx_shuffle)
        
            idx_query = idx_shuffle[:len(idx_shuffle)-250]
            idx_test = idx_shuffle[-250:]
            
            #Split data and thetas into test and training data
            data_test = data_query[idx_test]
            lags_test = lags_query[idx_test]
            theta_test  = theta_query[idx_test]
            
            data_query = data_query[idx_query]
            lags_query = lags_query[idx_query]
            theta_query = theta_query[idx_query]
            
            #save to disk
            saveData(data_query, generator.pars_conversion(theta_query), 
                     "data/locations/","active_locs_flux.csv", 
                     current_locs = pd.read_csv("data/locations/active_locs_flux.csv"))
            saveData(lags_query, generator.pars_conversion(theta_query), 
                     "data/locations/","active_locs_lags.csv", 
                     current_locs = pd.read_csv("data/locations/active_locs_lags.csv"),
                     lags = True)

            saveData(data_test, generator.pars_conversion(theta_test), 
                     "data/locations/",
                     "active_test_locs_flux.csv")
            saveData(lags_test, generator.pars_conversion(theta_test), 
                     "data/locations/",
                     "active_test_locs_lags.csv",
                     lags = True)

            del data_query, data_test, lags_test, lags_query, theta_query, theta_test
            
            # add rejected parameter sets back to original array for potential 
            # future use:
            theta_lhs = np.vstack([theta_lhs, theta_query_large[query_idx[n_samples:]]])
            
            # increment the index for reading parameters from theta_lhs
            lhs_idx += (n_samples_large)
    
            print("Setting up modeling")
            Xquery = FluxData("data/locations/active_locs_flux.csv", scaler, 
                                "active_scaler_flux.bin")
            print("Query data set created")
            flux_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            print("Query data loader created")
            
            Xquery = LagsData("data/locations/active_locs_lags.csv", scaler, 
                                "active_scaler_lags.bin")
            print("Query data set created")
            lags_dataloader = DataLoader(Xquery, batch_size=batch_size, 
                                          num_workers = num_workers, shuffle=True)
            print("Query data loader created")
            
            Xtest = FluxData("data/locations/active_test_locs_flux.csv", scaler, 
                               "active_scaler_flux.bin")
            print("Test data set created")
            flux_test_dataloader = DataLoader(Xtest, batch_size=batch_size,
                                         num_workers = num_workers, shuffle=True)
            print("Test data loader created")
            
            Xtest = LagsData("data/locations/active_test_locs_lags.csv", scaler, 
                               "active_scaler_lags.bin")
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
        torch.save(flux_model.state_dict(), "models/active_flux_final.pth")
        print("Saved PyTorch Model State to models/active_flux_final.pth")
        
        flux_tr_loss_arr = np.asarray(flux_tr_loss_arr)
        flux_te_loss_arr = np.asarray(flux_te_loss_arr)
        
        np.savetxt("loss/active_flux_te_loss.txt",flux_te_loss_arr)
        np.savetxt("loss/active_flux_tr_loss.txt",flux_tr_loss_arr)
        np.savetxt("loss/active_flux_epochs.txt",loop_flux_epochs)
        
        print("Final best lags training loss:", last_sig_lags_tr)
        print("Final best lags testing loss:", last_sig_lags_te)
        torch.save(lags_model.state_dict(), "models/active_flags_final.pth")
        print("Saved PyTorch Model State to models/active_lags_final.pth")
        
        lags_tr_loss_arr = np.asarray(lags_tr_loss_arr)
        lags_te_loss_arr = np.asarray(lags_te_loss_arr)
        
        np.savetxt("loss/active_lags_te_loss.txt",lags_te_loss_arr)
        np.savetxt("loss/active_lags_tr_loss.txt",lags_tr_loss_arr)
        np.savetxt("loss/active_lags_epochs.txt",loop_lags_epochs)

def grid(wrk_dir,device):
    print("Training using grid")
    rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
    rmf = unpack_rmf(rmf_name)
    egrid = rmf.e_min #energy grid used to evaluate the xspec model
    
    locations = "data/locations/"
    
    grid_sizes = [5,6,7,8,9,10]
    grid_names = []
    for size in grid_sizes:
        grid_names.append(f"grid_{size}")
    print(grid_names)
        
    batch_size = 128
    num_workers = 4
    
    modes = ["flux", "lags"]
    
    for (size,fname) in zip(grid_sizes,grid_names):
        print(f"Starting {size} x {size} grid loop")
        for mode in modes:
            print(f"Training on {mode}")
            
            scaler = MinMaxScaler()
            #create initial dataset object to create scaler (and then delete object)
            
            print("Dataloaders created")
            
            model = network.LightSharpNetwork(5,len(egrid))
            model.to(device)
            optimizer = Adam(model.parameters(),lr = 0.001)
            
            if mode == "flux":
                train = train_flux
                test = test_flux
                loss_fn = barredMSELoss(f"{fname}_{mode}_scaler.bin",device)
                dataType = FluxData
                
            elif mode == "lags":
                train = train_lags
                test = test_lags
                loss_fn = lagLoss(f"{fname}_{mode}_scaler.bin",device)
                dataType = LagsData
                
            else:
                print("Invalid mode, defaulting to flux modelling")
                train = train_flux
                test = test_flux
                loss_fn = barredMSELoss(f"{fname}_{mode}_scaler.bin",device)
                dataType = FluxData
                
            training_data = dataType(locations+f"loc_{fname}_{mode}.csv", scaler, 
                             scaler_name=f"{fname}_{mode}_scaler.bin", scaling=True)
        
            train_dataloader = DataLoader(training_data,batch_size=batch_size,
                                          num_workers = num_workers, shuffle=True)
            
            testing_data = dataType(locations+"loc_{fname}_{mode}_test.csv", scaler, 
                                 scaler_name=f"{fname}_{mode}_scaler.bin")
            
            test_dataloader = DataLoader(testing_data,batch_size=batch_size,
                                          num_workers = num_workers, shuffle=True)
                
            grid_training_loop(model, optimizer, train, test, 
                                   train_dataloader, test_dataloader,
                                   loss_fn, device,
                                   size, mode)
        
def main():
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
    print(device)
    
    queryByDropout(wrk_dir,device)
    grid(wrk_dir,device)

if __name__ == "__main__":
    main()
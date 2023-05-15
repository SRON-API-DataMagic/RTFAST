"""
This program serves to visualise a neural network's outputs vs the true values.
"""
from sherpa.astro.ui import unpack_rmf
import torch
import os


import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from joblib import load, Parallel, delayed
import scipy.stats

import pandas as pd
import seaborn as sns
from tqdm import tqdm

import network
from generator import lhs_trimmed_gen,pars_conversion,rtdist_flux
from main import CustomData, saveData

class LoadCustomData(CustomData):
    def __init__(self,pars,data,scaler):
        super().__init__(pars,data,scaler)
        self.par_list = pars
        self.data = data
        self.mask = np.where(self.data <= 1e-38, 0, 1)

    def __getitem__(self,idx):
        datum = self.data[idx]
        parameters = self.par_list[idx]
        mask = self.mask[idx]
        return datum, parameters, mask
    
class Residual():
    
    def __init__(self,residuals):
        self.data = residuals
            
def inverse(scaler,data):
    scaled_data = scaler.inverse_transform(data)
    return scaled_data

def residual_plots(egrid,pred,da,spin,mass,fname,title,gr,log = False, norm = False):
    fig, axs = plt.subplots(2,1,sharex=True)
    axs[0].plot(egrid,pred,c="blue",label="NN model")
    axs[0].plot(egrid,da,c="r",label="Truth",lw=1.)
    axs[0].legend()
    axs[0].text(0.3,0.5,f"Spin, Mass: {spin:>2f} {mass:>2f}",
                transform=axs[0].transAxes)
    axs[0].set_title(title)
    if log == True:
        axs[0].set_ylabel("Log(Flux)")
    else:
        axs[0].set_ylabel("Flux")
    axs[1].scatter(egrid,(da-pred)/da,s=0.5)
    axs[1].set_ylabel("Residuals")
    axs[1].set_xlabel("Energy in keV")
    axs[1].axhline(y=0.01,ls="--",color="orange")
    axs[1].axhline(y=-0.01,ls="--",color="orange")
    max_res = np.absolute((da-pred)/da).max()
    if max_res > 1 and norm == True:
        axs[1].set_ylim(-1,1)
    plt.savefig(f"samples/{gr}_{fname}.png")
    plt.close()

def loss_plots(gr):
    tr_loss_arr = np.loadtxt(f"loss/{gr}tr_loss.txt")
    te_loss_arr = np.loadtxt(f"loss/{gr}te_loss.txt")

    plt.plot(tr_loss_arr,label="training loss")
    plt.plot(te_loss_arr,label="testing loss")  
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"loss/{gr}loss_plot.png")
    plt.close()
    
def sample_dist_plots(pars):
    plt.hist(pars[:,0],bins=100)
    plt.xlabel("Spin")
    plt.ylabel("Number of samples")
    plt.savefig("sample_dist/spin_sample_dist.png")
    plt.close()
    
    plt.hist(pars[:,1],bins=100)
    plt.xlabel("Log(Mass)")
    plt.ylabel("Number of samples")
    plt.savefig("sample_dist/mass_sample_dist.png")
    plt.close()
    

def heatmap_plots(mass_res,mass_res_flat,spin_res,spin_res_flat,mass_tick, 
                  mass_ticklabel,spin_ticklabel,
                  spin_tick,gr):
    cmap_flat = sns.color_palette("hls", 2)
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(mass_res,cmap="vlag", vmin = 0, vmax = 0.05, center = 0.01)
    ax.set_yticks(mass_tick,labels=mass_ticklabel)
    ax.set_xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
    ax.set_xlabel("Energy in keV")
    ax.set_ylabel("Mass")
    plt.savefig(f"heatmaps/{gr}mass_hm.png")
    plt.close()
    print("Continuous mass hm plotted")
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(mass_res_flat,cmap=cmap_flat,cbar_kws = {})
    ax.set_yticks(mass_tick,labels=mass_ticklabel)
    ax.set_xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
    ax.set_xlabel("Energy in keV")
    ax.set_ylabel("Mass")
    colorbar = ax.collections[0].colorbar
    M=mass_res_flat.max().max()
    colorbar.set_ticks([1/4*M,3/4*M])
    colorbar.set_ticklabels(['< 1% error','> 1% error'])
    plt.savefig(f"heatmaps/{gr}flat_mass_hm.png")
    plt.close()
    print("Flat mass hm plotted")
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(spin_res,cmap="vlag", vmin = 0, vmax = 0.05, center = 0.01)
    ax.set_yticks(spin_tick,labels=spin_ticklabel)
    ax.set_xticks(np.arange(0,4096,4096/4),labels=np.arange(0,20,5))
    ax.set_xlabel("Energy in keV")
    ax.set_ylabel("Spin")
    plt.savefig(f"heatmaps/{gr}spin_hm.png")
    plt.close()
    print("Continuous spin hm plotted")
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(spin_res_flat,cmap=cmap_flat)
    ax.set_yticks(spin_tick,labels=spin_ticklabel)
    ax.set_xticks(np.arange(0,4096,4096/4),labels=np.arange(0,20,5))
    ax.set_xlabel("Energy in keV")
    ax.set_ylabel("Spin")
    colorbar = ax.collections[0].colorbar
    M=spin_res_flat.max().max()
    colorbar.set_ticks([1/4*M,3/4*M])
    colorbar.set_ticklabels(['< 1% error','> 1% error'])
    plt.savefig(f"heatmaps/{gr}flat_spin_hm.png")
    plt.close()
    print("Flat spin hm plotted")

def set_envir_vars(wrk_dir):
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

def retrieve_egrid(wrk_dir):
    rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
    rmf = unpack_rmf(rmf_name)
    egrid = rmf.e_min
    
    return egrid

def model_load(model_loc,egrid):
    model = network.NeuralNetwork(5,len(egrid))
    model.load_state_dict(torch.load(model_loc))
    model.eval()
    return model

def data_load(data_name):
    with open(f"data/{data_name}data.txt","r") as f1:
        data = np.loadtxt(f1)
    f1.close()
    
    with open(f"data/{data_name}pars.txt","r") as f2:
        pars = np.loadtxt(f2)
    f2.close()
    
    pars = pars[:,[1,13]]
    pars[:,1] = np.log10(pars[:,1])
    
    return data, pars

def pars_load(data_name):
    with open(f"data/{data_name}pars.txt","r") as f2:
        pars = np.loadtxt(f2)
    f2.close()
    
    pars = pars[:,[1,13]]
    pars[:,1] = np.log10(pars[:,1])
    
    return pars
    
def pars_comparison(previous_set,new_set):
    #redefine parameter lists as complex numbers
    complex_pre = previous_set[:,0] + previous_set[:,1]*1j
    complex_new = new_set[:,0] + new_set[:,1]*1j
    #find all matches
    mask = np.in1d(complex_new,complex_pre)
    #remove all data from new set already present in training data
    new_set = new_set[~mask]
    return new_set
    
def generate_test_set(size,egrid):
    """
    

    Parameters
    ----------
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    range_all = np.asarray(lhs_trimmed_gen())
    #pre generate Latin Hypercube samples.
    sampler = scipy.stats.qmc.LatinHypercube(d=len(range_all))
    sample = sampler.random(n=size)
    theta_lhs = scipy.stats.qmc.scale(sample, range_all[:,0], range_all[:,1])

    #generate physical models of test set
    theta_lhs_iterate = pars_conversion(theta_lhs)
    with Parallel(n_jobs=10,verbose=5) as parallel:
        #generate rtdist models for the correlated grid
        data_init = parallel(delayed(rtdist_flux)(pars, egrid)
                                        for pars in theta_lhs_iterate)
    data_init = np.asarray(data_init)
    print(data_init)
    return data_init, theta_lhs
    
def residual_computation(testing_dataloader,model,scaler):
    mass, spin = [], []
    residuals = []
    for batch, (D,P) in enumerate(testing_dataloader):
        spin.append(P[0][0].item())
        mass.append(10**P[0][1].item())
        pred = model(P).detach().numpy()
        pred = 10**(inverse(scaler,pred))
        resid = (D-pred)/D
        residuals.append(np.absolute(np.asarray(resid)))
        
    residuals = np.asarray(residuals)
    obj_residuals = []
    for res in residuals:
        obj_residuals.append(Residual(res))
    
    
    dataframe = pd.DataFrame({"Spin":spin,"Mass":mass,"Residuals":obj_residuals})
    dataframe.sort_values(by="Spin",inplace=True,ignore_index=True)
    
    del mass,spin,pred,residuals
    
    spins = np.zeros((len(dataframe)))
    resids_spin = np.zeros((len(dataframe),4096))
    resids_spin_flat = np.zeros((len(dataframe),4096))
    masses = np.zeros((len(dataframe)))
    
    for i,row in dataframe.iterrows():
        spins[i] = row["Spin"]
        resids_spin[i] = row["Residuals"].data
        resids_spin_flat[i] = np.where(row["Residuals"].data < 0.01, 0., 0.01 )
    
    spin_res = pd.DataFrame(resids_spin,index=spins)
    spin_res_flat = pd.DataFrame(resids_spin_flat,index=spins)
    
    resids_mass = np.zeros((len(dataframe),4096))
    resids_mass_flat = np.zeros((len(dataframe),4096))
    dataframe.sort_values(by="Mass",inplace=True,ignore_index=True)
    
    for i,row in dataframe.iterrows():
        resids_mass[i] = row["Residuals"].data
        masses[i] = row["Mass"]
        resids_mass_flat[i] = np.where(row["Residuals"].data < 0.01, 0., 0.01 )
        
    mass_res = pd.DataFrame(resids_mass,index=masses)
    mass_res_flat = pd.DataFrame(resids_mass_flat,index=masses)
    
    percents = [0,0.25,0.5,0.75]
    mass_tick = []
    mass_ticklabel = []
    spin_tick = []
    spin_ticklabel = []
    
    for p in percents:
        mass_tick.append(int(len(masses)*p))
        mass_ticklabel.append(f"{masses[int(len(masses)*p)]:.2E}")
        
        spin_tick.append(int(len(spins)*p))
        spin_ticklabel.append(f"{spins[int(len(spins)*p)]:.2f}")
        
    mass_tick.append(int(len(masses)-1))
    mass_ticklabel.append(f"{masses[int(len(masses)-1)]:.2E}")
    
    spin_tick.append(int(len(spins))-1)
    spin_ticklabel.append(f"{spins[int(len(spins))-1]:.2f}")
    
    return (mass_res,mass_res_flat,spin_res,spin_res_flat,mass_tick, 
                      mass_ticklabel,spin_ticklabel,
                      spin_tick)

def model_samples(testing_dataloader,scaler,model,egrid,gr):
    for batch, (D,P,M) in enumerate(testing_dataloader):
        #retrieve relevant data and parameters
        spin, mass = P[0][0].item(),10**P[0][1].item()
        da = torch.squeeze(D)
        
        #generate neural network prediction and rescale to linear space
        pred = model(P).detach().numpy()
        pred = 10**np.squeeze(inverse(scaler,pred))
        
        fname = f"{batch}_res"
        title = "Standard output"
        
        residual_plots(egrid, pred, da, spin, mass, fname, title, gr, norm = True)
        
        pred = model(P).detach().numpy()
        pred = np.squeeze(inverse(scaler,pred))
        
        da_log = np.log10(da)
        
        fname = f"{batch}_res_log"
        title = "Log scaled output"
    
        residual_plots(egrid, pred, da_log, spin, mass, fname, title, gr)
        
        pred = model(P).detach().numpy()
        pred = np.squeeze(pred)
        
        fname = f"{batch}_res_scal"
        title = "Neural network normalised output"
        
        da_log_scal = scaler.transform(da_log.reshape(1, -1)).flatten()
    
        residual_plots(egrid, pred, da_log_scal, spin, mass, fname, title, gr)
        
        if batch > 5:
            break

def calculate_loss(testing_dataloader,model,scaler):
    residuals = []
    for batch, (D,P,M) in enumerate(testing_dataloader):
        pred = model(P).detach().numpy()
        pred = 10**(inverse(scaler,pred))
        resid = (D-pred)/D
        resid = resid*M
        residuals.append(np.absolute(np.asarray(resid)))
    
    residuals = np.asarray(residuals)
    median = np.median(residuals)
    low_q, high_q = np.quantile(residuals,[0.25,0.75])
    return median,low_q,high_q

def active_v_grid(wrk_dir,egrid,testing_dataloader,active_scaler,grid_scaler):
    model_base_loc = wrk_dir+"/models/"
    
    active_name = [0,1,2,3,4,5,10,15,19]
    active_model_names = np.array([0,1,2,3,4,5,10,15,19])
    active_sample_nums = (active_model_names+2)*5000
    active_model_names = [model_base_loc+str(i)+"_model.pth" for i in active_model_names]
    grid_name = [5,6,7,8,9,10]
    grid_model_names = np.array([5,6,7,8,9,10])
    grid_sample_nums = grid_model_names**5
    grid_model_names = [model_base_loc+"grid_"+str(i)+".pth" for i in grid_model_names]
    
    active_median_loss = []
    active_loss_low_q = []
    active_loss_high_q = []
    grid_median_loss = []
    grid_loss_low_q = []
    grid_loss_high_q = []
    
    print("Calculating loss for active learning")
    for (model_loc,gr) in zip(active_model_names,active_name):
        gr = str(gr) + "_active"
        print(gr)
        model = model_load(model_loc, egrid)
        model.eval()
        median,low_q,high_q = calculate_loss(testing_dataloader, model, active_scaler)
        active_median_loss.append(median)
        active_loss_low_q.append(low_q)
        active_loss_high_q.append(high_q)
        #model_samples(testing_dataloader,active_scaler,model,egrid,gr)
        """
        (mass_res,mass_res_flat,spin_res,spin_res_flat,mass_tick, 
                          mass_ticklabel,spin_ticklabel,
                          spin_tick) = residual_computation(testing_dataloader, model, grid_scaler)
        heatmap_plots(mass_res,mass_res_flat,spin_res,spin_res_flat,mass_tick, 
                          mass_ticklabel,spin_ticklabel,
                          spin_tick,gr)
        """
    
    active_median_loss = np.asarray(active_median_loss)
    active_loss_low_q = np.asarray(active_loss_low_q)
    active_loss_high_q = np.asarray(active_loss_high_q)
    
    print("Calculating loss for grid learning")
    for (model_loc,gr) in zip(grid_model_names,grid_name):
        gr = str(gr) + "_grid"
        print(gr)
        model = model_load(model_loc, egrid)
        model.eval()
        median,low_q,high_q = calculate_loss(testing_dataloader, model, grid_scaler)
        grid_median_loss.append(median)
        grid_loss_low_q.append(low_q)
        grid_loss_high_q.append(high_q)
        #model_samples(testing_dataloader,grid_scaler,model,egrid,gr)
        """
        (mass_res,mass_res_flat,spin_res,spin_res_flat,mass_tick, 
                          mass_ticklabel,spin_ticklabel,
                          spin_tick) = residual_computation(testing_dataloader, model, grid_scaler)
        heatmap_plots(mass_res,mass_res_flat,spin_res,spin_res_flat,mass_tick, 
                          mass_ticklabel,spin_ticklabel,
                          spin_tick,gr)
        """
    
    grid_median_loss = np.asarray(grid_median_loss)
    grid_loss_low_q = np.asarray(grid_loss_low_q)
    grid_loss_high_q = np.asarray(grid_loss_high_q)
    
    print("Plotting loss by sample size")
    
    plt.fill_between(grid_sample_nums, grid_median_loss+grid_loss_high_q, 
                     grid_median_loss-grid_loss_low_q, alpha = 0.5,color = "orange",
                     zorder=1)
    plt.plot(grid_sample_nums,grid_median_loss,label="Grid",color = "orange",
             zorder=1)
    plt.fill_between(active_sample_nums, active_median_loss+active_loss_high_q, 
                     active_median_loss-active_loss_low_q, alpha = 0.5,color = "blue",
                     zorder=2)
    plt.plot(active_sample_nums,active_median_loss,label="Active learning",
             color = "blue",zorder=2)
    plt.axhline(y=1e-2, ls = "--",label="1% error",zorder=3,color="green")
    plt.yscale("log")
    plt.xlabel("Number of samples used in training")
    plt.ylabel("Average percentage error")
    plt.legend()
    plt.savefig("loss/loss_by_sample_size.png")
    plt.close()

def main():
    wrk_dir = os.getcwd()
    
    set_envir_vars(wrk_dir)
    
    #retrieve scalers
    active_scaler = MinMaxScaler()
    active_scaler = load('scalers/active_scaler.bin')
    grid_scaler = MinMaxScaler()
    grid_scaler = load('scalers/grid_scaler.bin')
    
    egrid = retrieve_egrid(wrk_dir)
    
    with open(f"data/test_data.txt","r") as f1:
        test_data = np.loadtxt(f1)
    f1.close()
    
    with open(f"data/test_pars.txt","r") as f1:
        test_pars = np.loadtxt(f1)
    f1.close()
    
    test_pars[:,13] = np.log10(test_pars[:,13]) 
    test_pars[:,2] = test_pars[:,2]
    test_pars[:,3] = test_pars[:,3]
    test_pars[:,4] = np.log10(test_pars[:,4])
    test_pars = test_pars[:,[1,13,2,3,4]] #retrieve parameters
    
    print(test_pars)
    print(test_data)
    
    data = test_data
    pars = test_pars
    
    #put test set into dataloader format
    batch_size = 1
    test_data = LoadCustomData(pars,data,grid_scaler) #scaler unused but must be parsed
    testing_dataloader = DataLoader(test_data,batch_size = batch_size)
    
    active_v_grid(wrk_dir,egrid,testing_dataloader,active_scaler,grid_scaler)
    
if __name__ == "__main__":
    main()
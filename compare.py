"""
This program serves to visualise a neural network's outputs vs the true values.
"""
from sherpa.astro.ui import unpack_rmf
import torch
import os
import time

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
    
    def __init__(self,residuals,flat):
        self.data = residuals
        self.flat = flat
            
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
    
def flat_heatmap(df,index,ticks,ticklabels,fname):
    cmap_flat = sns.color_palette("hls", 2)
    
    residuals= []
    for indice,row in df.iterrows():
        residuals.append(row['Residuals'].flat)
    residuals = np.squeeze(np.asarray(residuals))
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(residuals,cmap=cmap_flat,cbar_kws = {})
    ax.set_yticks(ticks,labels=ticklabels)
    ax.set_xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
    ax.set_xlabel("Energy in keV")
    ax.set_ylabel("Mass")
    colorbar = ax.collections[0].colorbar
    maxi=df[index].max()
    colorbar.set_ticks([1/4*maxi,3/4*maxi])
    colorbar.set_ticklabels(['< 1% error','> 1% error'])
    plt.savefig(f"heatmaps/{fname}_flat_{index}.png")
    plt.close()
    print(f"Flat {fname} hm plotted")
    return

def continuous_heatmap(df,index,ticks,ticklabels,fname):
    residuals,indexes = [],[]
    for indice,row in df.iterrows():
        residuals.append(row['Residuals'].data)
    residuals = np.squeeze(np.asarray(residuals))
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(residuals,cmap="vlag", vmin = 0, vmax = 0.05, center = 0.01)
    ax.set_yticks(ticks,labels=ticklabels)
    ax.set_xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
    ax.set_xlabel("Energy in keV")
    ax.set_ylabel("Mass")
    plt.savefig(f"heatmaps/{fname}_{index}_hm.png")
    plt.close()
    print(f"Continuous {fname} hm plotted")
    return

def heatmap_plots(df, indexes, ticks, ticklabels, fname):
    for i,index in enumerate(indexes):
        print(index)
        df.sort_values(by=index,inplace=True,ignore_index=True)
        continuous_heatmap(df, index, ticks[i], ticklabels[i], fname)
        flat_heatmap(df, index, ticks[i], ticklabels[i], fname)

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
    return data_init, theta_lhs
    

def residual_sorting(df,indexing):
    df.sort_values(by=indexing,inplace=True,ignore_index=True)
    percents = [0,0.25,0.5,0.75]
    tick = []
    ticklabel = []
    for p in percents:
        tick.append(int(len(df)*p))
        ticklabel.append(f"{df[indexing][int(len(df)*p)]:.2E}")
    return tick,ticklabel

def residual_computation(testing_dataloader,model,scaler):
    mass, spin, inc, rin, rout = [], [], [], [], []
    residuals,residuals_flat = [], []
    for batch, (D,P,M) in enumerate(testing_dataloader):
        spin.append(P[0][0].item())
        mass.append(10**P[0][1].item())
        inc.append(P[0][2].item())
        rin.append(P[0][3].item())
        rout.append(10**P[0][4].item())
        pred = model(P).detach().numpy()
        pred = 10**(inverse(scaler,pred))
        resid = (D-pred)/D
        resid[M == 0] = 0
        resid = np.absolute(np.asarray(resid))
        resid_flat = np.where(np.absolute(np.asarray(resid)) < 0.01, 0., 0.01 )
        residuals.append(resid)
        residuals_flat.append(resid_flat)
        
    residuals = np.asarray(residuals)
    residuals_flat = np.asarray(residuals_flat)
    obj_residuals = []
    for (res,res_flat) in zip(residuals,residuals_flat):
        obj_residuals.append(Residual(res,res_flat))

    dataframe = pd.DataFrame({"Spin":spin,"Mass":mass,"Inclination":inc,
                              "Inner R":rin,"Outer R":rout,
                              "Residuals":obj_residuals})
    del mass,spin,inc,rin,rout,pred,residuals
    
    mass_tick, mass_ticklabel = residual_sorting(dataframe, "Mass")
    spin_tick, spin_ticklabel = residual_sorting(dataframe, "Spin")
    inc_tick, inc_ticklabel = residual_sorting(dataframe, "Inclination")
    rin_tick, rin_ticklabel = residual_sorting(dataframe, "Inner R")
    rout_tick, rout_ticklabel = residual_sorting(dataframe, "Outer R")
    
    ticks = [mass_tick,spin_tick,inc_tick,rin_tick,rout_tick]
    ticklabels = [mass_ticklabel,spin_ticklabel,inc_ticklabel,rin_ticklabel,
                  rout_ticklabel]
    
    return (dataframe, ticks, ticklabels)

def model_samples(testing_dataloader,scaler,model,egrid,gr):
    for batch, (D,P,M) in enumerate(testing_dataloader):
        M = np.squeeze(M)
        D = np.squeeze(D)
        #retrieve relevant data and parameters
        spin, mass, inc, rin, rout = (P[0][0].item(),10**P[0][1].item(),
                                      P[0][2].item(),P[0][3].item(),
                                      10**P[0][4].item())
        D[M==0] = 1e-38
        da = np.squeeze(D)
        
        #generate neural network prediction and rescale to linear space
        pred = model(P).detach().numpy()
        pred = 10**np.squeeze(inverse(scaler,pred))
        pred[M==0] = 1e-38
        
        fname = f"{batch}_res"
        title = "Standard output"
        
        residual_plots(egrid, pred, da, spin, mass, fname, title, gr, norm = True)
        
        da_log = np.log10(da)
        pred = model(P).detach().numpy()
        pred = np.squeeze(inverse(scaler,pred))
        
        fname = f"{batch}_res_log"
        title = "Log scaled output"
    
        residual_plots(egrid, pred, da_log, spin, mass, fname, title, gr)
        
        da_log_scal = scaler.transform(da_log.reshape(1, -1)).flatten()
        
        pred = model(P).detach().numpy()
        pred = np.squeeze(pred)
        
        fname = f"{batch}_res_scal"
        title = "Neural network normalised output"
    
        residual_plots(egrid, pred, da_log_scal, spin, mass, fname, title, gr)
        
        if batch > 5:
            break

def calculate_loss(testing_dataloader,model,scaler):
    residuals = []
    for batch, (D,P,M) in enumerate(testing_dataloader):
        pred = model(P).detach().numpy()
        pred = 10**(inverse(scaler,pred))
        resid = (D-pred)/D
        resid[M == 0] = 0
        residuals.append(np.absolute(np.asarray(resid)))
    
    residuals = np.asarray(residuals)
    return residuals

def violin(df,fname):
    sns.violinplot(data=df, x="Sample Size", y="Residuals")
    plt.ylim(top=1)
    plt.savefig(f"loss/violin_{fname}.png")
    plt.close()

def box(df,fname):
    sns.boxplot(data=df, x="Sample Size", y="Residuals",whis=1.8)
    plt.ylim(top=1)
    plt.savefig(f"loss/box_{fname}.png")
    plt.close()

def residuals_dataframe(residuals,names):
    d = {"Residuals":[],"Sample Size":[]}
    for i,name in enumerate(names):
        print(name)
        for point in residuals[i].flatten():
            d["Residuals"].append(point)
            d["Sample Size"].append(name) 
    print("# of residuals:"+str(len(d["Residuals"])))
    print("# of labels:"+str(len(d["Sample Size"])))
    time_start = time.time()
    df = pd.DataFrame(data = d)
    print(time.time()-time_start)
    return df

def active_v_grid(wrk_dir,egrid,testing_dataloader,active_scaler,grid_scaler):
    model_base_loc = wrk_dir+"/models/save/"
    
    indexes = ["Mass", "Spin", "Inclination", "Inner R", "Outer R"]
    
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
    resid_list = []
    
    print("Calculating loss for active learning")
    for (model_loc,fname) in zip(active_model_names,active_name):
        fname = str(fname) + "_active"
        print(fname)
        model = model_load(model_loc, egrid)
        model.eval()
        residuals = calculate_loss(testing_dataloader, model, active_scaler)
        resid_list.append(residuals)
        median = np.median(residuals)
        low_q, high_q = np.quantile(residuals,[0.25,0.75])
        active_median_loss.append(median)
        active_loss_low_q.append(low_q)
        active_loss_high_q.append(high_q)
        model_samples(testing_dataloader,active_scaler,model,egrid,fname)
        df, ticks, ticklabels = residual_computation(testing_dataloader, model, active_scaler)
        heatmap_plots(df, indexes, ticks, ticklabels, fname)
        del df, ticks, ticklabels
    resid_list = np.asarray(resid_list)
    df = residuals_dataframe(resid_list, active_sample_nums)
    time_start = time.time()
    violin(df,"active")
    print("Violin plot render time:"+str(time.time()-time_start))
    time_start = time.time()
    box(df,"active")
    print("Box plot render time:"+str(time.time()-time_start))
    
    del df
    
    active_median_loss = np.asarray(active_median_loss)
    active_loss_low_q = np.asarray(active_loss_low_q)
    active_loss_high_q = np.asarray(active_loss_high_q)
    
    resid_list = []
    
    print("Calculating loss for grid learning")
    for (model_loc,fname) in zip(grid_model_names,grid_name):
        fname = str(fname) + "_grid"
        print(fname)
        model = model_load(model_loc, egrid)
        model.eval()
        residuals = calculate_loss(testing_dataloader, model, grid_scaler)
        resid_list.append(residuals)
        median = np.median(residuals)
        low_q, high_q = np.quantile(residuals,[0.25,0.75])
        grid_median_loss.append(median)
        grid_loss_low_q.append(low_q)
        grid_loss_high_q.append(high_q)
        model_samples(testing_dataloader,grid_scaler,model,egrid,fname)
        df, ticks, ticklabels = residual_computation(testing_dataloader, model, 
                                                     grid_scaler)
        heatmap_plots(df, indexes, ticks, ticklabels, fname)
        del df, ticks, ticklabels
    
    resid_list = np.asarray(resid_list)
    df = residuals_dataframe(resid_list, grid_sample_nums)
    time_start = time.time()
    violin(df,"grid")
    print("Violin plot render time:"+str(time.time()-time_start))
    time_start = time.time()
    box(df,"grid")
    print("Box plot render time:"+str(time.time()-time_start))
    
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
    """
    data, pars = generate_test_set(5000, egrid)
    
    np.savetxt("data/test_data.txt",data)
    np.savetxt("data/test_pars.txt",pars_conversion(pars))
    """
    with open("data/test_data.txt","r") as f1:
        test_data = np.loadtxt(f1)
    f1.close()
    
    with open("data/test_pars.txt","r") as f1:
        test_pars = np.loadtxt(f1)
    f1.close()
    
    test_pars[:,13] = np.log10(test_pars[:,13]) 
    test_pars[:,2] = test_pars[:,2]
    test_pars[:,3] = test_pars[:,3]
    test_pars[:,4] = np.log10(test_pars[:,4])
    test_pars = test_pars[:,[1,13,2,3,4]] #retrieve parameters
    
    data = test_data
    pars = test_pars
    
    #put test set into dataloader format
    batch_size = 1
    test_data = LoadCustomData(pars,data,grid_scaler) #scaler unused but must be parsed
    testing_dataloader = DataLoader(test_data,batch_size = batch_size)
    
    active_v_grid(wrk_dir,egrid,testing_dataloader,active_scaler,grid_scaler)
    
if __name__ == "__main__":
    main()
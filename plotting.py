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
from generator import lhs_trimmed_gen, pars_conversion, rtdist_flux, rtdist_lags
from dataStructures import FluxData, LagsData
from processing import saveData, nanChecker

class LoadFluxData(FluxData):
    def __init__(self,labels,scaler,scaler_name):
        super().__init__(labels,scaler,scaler_name)
        
    def __getitem__(self,idx):
        #retrieve location of the spectra to load
        location = self.labels.iloc[idx,-1]
        #retrieve parameters used to generate the spectra that we want to train on
        parameters = self.labels.iloc[idx,self.pars_list].astype(float)
        #convert parameters to log space
        parameters.iloc[[3]] = -parameters.iloc[[3]]
        parameters.iloc[[1,2,3,4]] = np.log10(parameters.iloc[[1,2,3,4]])
        parameters = torch.tensor(parameters)
        #load spectra
        datum = np.loadtxt(location).reshape(1, -1)
        try:
            mask = np.where(datum <= 1e-38, 0, 1)
        except:
            mask = None
        if mask is not None:
            return datum, parameters, mask
        else:
            return datum, parameters

class LoadLagsData(LagsData):
    def __init__(self,labels,scaler,scaler_name):
        super().__init__(labels,scaler,scaler_name)
        
    def __getitem__(self,idx):
        #retrieve location of the spectra to load
        location = self.labels.iloc[idx,-1]
        #retrieve parameters used to generate the spectra that we want to train on
        parameters = self.labels.iloc[idx,self.pars_list].astype(float)
        #convert parameters to log space
        parameters.iloc[[3]] = -parameters.iloc[[3]]
        parameters.iloc[[1,2,3,4]] = np.log10(parameters.iloc[[1,2,3,4]])
        parameters = torch.tensor(parameters)
        #load spectra
        datum = np.loadtxt(location).reshape(1, -1)
        return datum, parameters

class Residual():
    
    def __init__(self,residuals,flat):
        self.data = residuals
        self.flat = flat

class Losses():
    
    def set_loss(self,residuals,name):
        super().__setattr__(name, residuals)
        
            
def inverse(scaler,data):
    scaled_data = scaler.inverse_transform(data)
    return scaled_data

def returnFlats(df):
    return df.flat

def returnContinuous(df):
    return df.data

def distributions(data,labels,fname):
    for i,column in enumerate(data.T):
        plt.hist(column, bins=100, density = True)
        plt.xlabel(labels[i])
        plt.savefig(fname+labels[i]+".png")
        plt.close()

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
    
def flat_heatmap(df,index,ticks,ticklabels,fname):
    colors = ["#16E6E9", "#E91916"]
    cmap_flat = sns.color_palette(colors)
    
    residuals = df["Residuals"].apply(returnFlats)
    resids = []
    for item in residuals:
        resids.append(item)
    resids = np.asarray(resids)
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(resids,cmap=cmap_flat,cbar_kws = {})
    ax.set_yticks(ticks,labels=ticklabels)
    ax.set_xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
    ax.set_xlabel("Energy in keV")
    ax.set_ylabel(index)
    colorbar = ax.collections[0].colorbar
    maxi=1
    colorbar.set_ticks([1/4*maxi,3/4*maxi])
    colorbar.set_ticklabels(['< 1% error','> 1% error'])
    plt.savefig(f"heatmaps/{fname}_flat_{index}.png")
    plt.close()
    print(f"Flat {fname} hm plotted")
    return

def continuous_heatmap(df,index,ticks,ticklabels,fname):
    residuals = df["Residuals"].apply(returnContinuous)
    resids = []
    for item in residuals:
        resids.append(item)
    resids = np.asarray(resids)
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(resids,cmap="vlag", vmin = 0, vmax = 0.05, center = 0.01)
    ax.set_yticks(ticks,labels=ticklabels)
    ax.set_xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
    ax.set_xlabel("Energy in keV")
    ax.set_ylabel(index)
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

def model_load(model_loc,egrid, lags = None):
    if lags == None:
        model = network.SharpNetwork(5,len(egrid))
    else:
        model = network.LagsNetwork(5,len(egrid))
    model.load_state_dict(torch.load(model_loc))
    model.eval()
    return model
    
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
        lags = parallel(delayed(rtdist_lags)(pars, egrid)
                                        for pars in theta_lhs_iterate)
    data_init = np.asarray(data_init)
    lags = np.asarray(lags)
    return data_init, lags, theta_lhs_iterate
    
def residual_sorting(df,indexing):
    df.sort_values(by=indexing,inplace=True,ignore_index=True)
    percents = [0,0.25,0.5,0.75]
    tick = []
    ticklabel = []
    for p in percents:
        tick.append(int(len(df)*p))
        ticklabel.append(f"{df[indexing][int(len(df)*p)]:.2E}")
    return tick,ticklabel

def residual_computation(testing_dataloader, model, scaler, mode):
    mass, spin, inc, rin, rout = [], [], [], [], []
    residuals,residuals_flat = [], []
    if mode == "flux":
        for batch, (D,P,M) in enumerate(tqdm(testing_dataloader)):
            spin.append(P[0][0].item())
            mass.append(10**P[0][1].item())
            inc.append(P[0][2].item())
            rin.append(P[0][3].item())
            rout.append(P[0][4].item())
            pred = model(P).detach().numpy()
            pred = 10**(inverse(scaler,pred))
            resid = (D-pred)/D
            resid[M == 0] = 0
            resid = np.absolute(np.asarray(resid))
            resid_flat = np.where(np.absolute(np.asarray(resid)) < 0.01, 0., 1. )
            residuals.append(resid)
            residuals_flat.append(resid_flat)
    else:
        for batch, (D,P) in enumerate(tqdm(testing_dataloader)):
            spin.append(P[0][0].item())
            mass.append(10**P[0][1].item())
            inc.append(P[0][2].item())
            rin.append(P[0][3].item())
            rout.append(P[0][4].item())
            pred, I_pred = model(P)
            pred = 10**(inverse(scaler,pred.detach().numpy()))
            pred = pred*np.where(I_pred.detach().numpy() > 0.5, 1, -1)
            resid = (D-pred)/D
            resid[np.abs(D)<1e-6] = 0
            resid = np.absolute(np.asarray(resid))
            resid_flat = np.where(np.absolute(np.asarray(resid)) < 0.01, 0., 1. )
            residuals.append(resid)
            residuals_flat.append(resid_flat)
        
    residuals = np.squeeze(np.asarray(residuals))
    residuals_flat = np.squeeze(np.asarray(residuals_flat))
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

def model_samples(testing_dataloader,scaler,model,egrid,gr,mode):
    if mode == "flux":
        for batch, (D,P,M) in enumerate(testing_dataloader):
            M = np.squeeze(M)
            D = np.squeeze(D)
            #retrieve relevant data and parameters
            spin, mass, inc, rin, rout = (P[0][0].item(),10**P[0][1].item(),
                                          10**P[0][2].item(),-10**P[0][3].item(),
                                          10**P[0][4].item())
            D[M==0] = 1e-38
            da = np.squeeze(D)
            
            #generate neural network prediction and rescale to linear space
            pred = model(P).detach().numpy()
            pred = 10**np.squeeze(inverse(scaler,pred))
            pred[M==0] = 1e-38
            
            fname = f"{batch}_{mode}_res"
            title = "Standard output"
            
            residual_plots(egrid, pred, da, spin, mass, fname, title, gr, norm = True)
            
            da_log = np.log10(da)
            pred = model(P).detach().numpy()
            pred = np.squeeze(inverse(scaler,pred))
            
            fname = f"{batch}_{mode}_res_log"
            title = "Log scaled output"
        
            residual_plots(egrid, pred, da_log, spin, mass, fname, title, gr)
            
            da_log_scal = scaler.transform(da_log.reshape(1, -1)).flatten()
            
            pred = model(P).detach().numpy()
            pred = np.squeeze(pred)
            
            fname = f"{batch}_{mode}_res_scal"
            title = "Neural network normalised output"
        
            residual_plots(egrid, pred, da_log_scal, spin, mass, fname, title, gr)
            
            if batch > 5:
                break
    else:
        for batch, (D,P) in enumerate(testing_dataloader):
            D = np.squeeze(D)
            #retrieve relevant data and parameters
            spin, mass, inc, rin, rout = (P[0][0].item(),10**P[0][1].item(),
                                          10**P[0][2].item(),-10**P[0][3].item(),
                                          10**P[0][4].item())
            D[(D<0)&(np.abs(D)<1e-6)] = -1e-6
            D[(D>0)&(D<1e-6)] = 1e-6
            da = np.squeeze(D)
            
            #generate neural network prediction and rescale to linear space
            pred, I_pred = model(P).detach().numpy()
            pred = 10**np.squeeze(inverse(scaler,pred))
            pred[pred<1e-6] = 1e-6
            pred = pred*np.where(I_pred > 0.5, 1, -1)
            
            fname = f"{batch}_{mode}_res"
            title = "Standard output"
            
            residual_plots(egrid, pred, da, spin, mass, fname, title, gr, norm = True)
            
            da_log = np.log10(np.abs(da))
            pred = model(P).detach().numpy()
            pred = np.squeeze(inverse(scaler,pred))
            
            fname = f"{batch}_{mode}_res_log"
            title = "Log scaled output"
        
            residual_plots(egrid, pred, da_log, spin, mass, fname, title, gr)
            
            da_log_scal = scaler.transform(da_log.reshape(1, -1)).flatten()
            
            pred = model(P).detach().numpy()
            pred = np.squeeze(pred)
            
            fname = f"{batch}_{mode}_res_scal"
            title = "Neural network normalised output"
        
            residual_plots(egrid, pred, da_log_scal, spin, mass, fname, title, gr)
            
            if batch > 5:
                break

def calculate_loss(testing_dataloader,model,scaler, mode = "flux"):
    residuals = []
    if mode == "flux":
        for batch, (D,P,M) in enumerate(tqdm(testing_dataloader)):
            pred = model(P).detach().numpy()
            pred = 10**(inverse(scaler,pred))
            resid = (D-pred)/D
            resid[D < 1e-38] = 0
            residuals.append(np.absolute(np.asarray(resid)))
    else:
        for batch, (D,P) in enumerate(tqdm(testing_dataloader)):
            pred, I_pred = model(P)
            pred = 10**(inverse(scaler,pred.detach().numpy()))
            pred = pred*np.where(I_pred.detach().numpy() > 0.5, 1, -1)
            resid = (D-pred)/D
            resid[np.abs(D)<1e-6] = 0
            residuals.append(np.absolute(np.asarray(resid)))
    residuals = np.asarray(residuals)
    return residuals

def violin(df,fname):
    sns.violinplot(data=df, x="Sample Size", y="Residuals")
    plt.ylim(top=1)
    plt.axhline(y=0.01,ls="--",color="orange")
    plt.savefig(f"loss/violin_{fname}.png")
    plt.close()

def box(df,fname):
    sns.boxplot(data=df, x="Sample Size", y="Residuals",whis=1.8)
    plt.ylim(top=1)
    plt.axhline(y=0.01,ls="--",color="orange")
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

def loss_epochs_plot(loss_base_loc,mode):
    
    train_names = [loss_base_loc+f"grid_{i}_{mode}_tr_loss.txt" for i in range(5,11)]
    test_names = [loss_base_loc+f"grid_{i}_{mode}_te_loss.txt" for i in range(5,11)]
    
    active_loss = np.loadtxt(loss_base_loc+f"30_{mode}_tr_loss.txt")
    active_test = np.loadtxt(loss_base_loc+f"30_{mode}_te_loss.txt")

    plt.plot(np.loadtxt(train_names[-1]), label="Training loss: 10x10 grid", 
             c = "red", ls = "-")
    plt.plot(np.loadtxt(test_names[-1]), label = "Validation loss: 10x10 grid", 
             c = "red", ls = "--")
    
    plt.plot(active_loss,label = "Training loss: active learning", c = "blue",
             ls = "-")
    plt.plot(active_test,label = "Validation loss: active learning", c = "blue",
             ls = "--")
    
    plt.yscale("log")
    plt.xlabel("Training epochs")
    plt.ylabel("Loss")
    plt.title("Comparison of loss by strategy")
    plt.legend()
    plt.savefig("loss/loss_over_time.png")
    plt.close()

def analysis(names, locs, nums, scaler_names, egrid, lags = None):
    if lags != None:
        mode = "lags"
    else:
        mode = "flux"
    scaler_base_loc = os.getcwd()+"/scalers/"
    indexes = ["Mass", "Spin", "Inclination", "Inner R", "Outer R"]
    
    median_loss = []
    loss_01_q = []
    loss_05_q = []
    loss_25_q = []
    loss_75_q = []
    loss_95_q = []
    loss_99_q = []
    loss_low_out = []
    loss_high_out = []
    
    resid_list = []
    
    if type(scaler_names) != list:
        tmp = [scaler_names for i in range(len(names))]
        scaler_names = tmp
        
    for (model_loc,fname,scaler_name) in zip(locs, names, scaler_names):
        scaler = load(scaler_base_loc+scaler_name)
        #put test set into dataloader format
        batch_size = 1
        if lags == None:
            print("Loading flux data")
            test_data = LoadFluxData("data/locations/loc_flux_test.csv",scaler,
                                       scaler_name) #scaler unused but must be parsed
        else:
            print("Loading lag data")
            test_data = LoadLagsData("data/locations/loc_lags_test.csv",scaler,
                                       scaler_name) #scaler unused but must be parsed
        testing_dataloader = DataLoader(test_data,batch_size = batch_size,
                                        num_workers=4)
        
        #folname = str(fname)
        fname = str(fname)
        print(fname)
        if lags != None:
            model = model_load(model_loc, egrid[:-1], lags = lags)
        else:
            model = model_load(model_loc, egrid, lags = lags)
        residuals = calculate_loss(testing_dataloader, model, scaler, mode)
        resid_list.append(residuals)
        median = np.median(residuals)
        q_01,q_05, q_25, q_75, q_95, q_99 = np.quantile(residuals,
                                                        [0.01,0.05,0.25,0.75,
                                                         0.95,0.99])
        median_loss.append(median)
        loss_05_q.append(q_05)
        loss_25_q.append(q_25)
        loss_75_q.append(q_75)
        loss_95_q.append(q_95)
        loss_99_q.append(q_99)
        loss_01_q.append(q_01)
        high_outliers = residuals[residuals >= np.percentile(residuals, 99)][::1000]
        low_outliers = residuals[residuals <= np.percentile(residuals, 1)][::1000]
        loss_low_out.append(low_outliers)
        loss_high_out.append(high_outliers)
        model_samples(testing_dataloader, scaler, model, egrid, fname, mode)
        df, ticks, ticklabels = residual_computation(testing_dataloader, 
                                                     model, scaler, mode)
        heatmap_plots(df, indexes, ticks, ticklabels, fname)
        #energy_plots(testing_dataloader, scaler, model, egrid, fname, folname)
        del df, ticks, ticklabels
        
    resid_list = np.asarray(resid_list)
    df = residuals_dataframe(resid_list, nums)
    print(df["Sample Size"].max())
    large_resid = df[df["Sample Size"] == df["Sample Size"].max()].max()
    print(f"The largest residual was {large_resid}")
    over = len(df[(df["Sample Size"] == df["Sample Size"].max())&(df["Residuals"] >= 0.01)])
    print(f"There were {over} residuals over 1%")
    overall = len(df[df["Sample Size"] == df["Sample Size"].max()])
    print(f"There were {overall} residuals overall")
    print(f"The ratio of residuals over 1% was {over/overall}")
    time_start = time.time()
    violin(df,"active")
    print("Violin plot render time:"+str(time.time()-time_start))
    time_start = time.time()
    box(df,"active")
    print("Box plot render time:"+str(time.time()-time_start))
    
    quantiles = [loss_01_q, loss_05_q, loss_25_q,
                 loss_75_q, loss_95_q, loss_99_q,
                 loss_high_out, loss_low_out, median_loss]
    quantile_names = ["q_1", "q_5", "q_25", "q_75", "q_95", "q_99", "high", 
                      "low", "median"]
    quants = Losses()
    for quant_arr, quant_name in zip(quantiles,quantile_names):
        quants.set_loss(quant_arr,quant_name)
    
    return quants
    
def active_v_grid(wrk_dir, egrid, lags_egrid):
    model_base_loc = wrk_dir+"/models/"
    loss_base_loc = wrk_dir+"/loss/"
    
    active_name = [0,1,2,3,4,10,15,20,25,30]
    active_name = np.array(active_name)
    active_sample_flux_nums = []
    active_sample_lags_nums = []
    for name in active_name:
        active_sample_flux_nums.append(len(pd.read_csv(f"data/locations/loc_flux_{name}.csv")))
        active_sample_lags_nums.append(len(pd.read_csv(f"data/locations/loc_lags_{name}.csv")))
    active_flux_names = [model_base_loc+str(i)+"_flux_model.pth" for i in active_name]
    active_lags_names = [model_base_loc+str(i)+"_lags_model.pth" for i in active_name]
    grid_flux_name = [f"grid_{i}_flux" for i in range(5,11)]
    grid_lags_name = [f"grid_{i}_lags" for i in range(5,11)]
    grid_flux_scaler = [f"grid_{i}_flux_scaler.bin" for i in range(5,11)]
    grid_lags_scaler = [f"grid_{i}_lags_scaler.bin" for i in range(5,11)]
    active_flux_scaler = "active_scaler_flux.bin"
    active_lags_scaler = "active_scaler_lags.bin"
    grid_model_names = np.array([5,6,7,8,9,10])
    grid_sample_nums = grid_model_names**5
    grid_model_flux_names = [model_base_loc+f"grid_{i}_flux.pth" for i in grid_model_names]
    grid_model_lag_names = [model_base_loc+f"grid_{i}_lags.pth" for i in grid_model_names]
    
    loss_epochs_plot(loss_base_loc,"flux")
    loss_epochs_plot(loss_base_loc,"lags")
    
    grid_flux = analysis(grid_flux_name, grid_model_flux_names, grid_sample_nums,
                            grid_flux_scaler, egrid)
    
    grid_lags = analysis(grid_lags_name, grid_model_lag_names, grid_sample_nums,
                            grid_lags_scaler, lags_egrid, lags=True)
    
    active_flux = analysis(active_name, active_flux_names, active_sample_flux_nums, 
                      active_flux_scaler, egrid)
    
    active_lags = analysis(active_name, active_lags_names, active_sample_lags_nums, 
                      active_lags_scaler, lags_egrid, lags=True)
    
    
    print("Plotting loss by sample size")
    print("Plotting fluxes")
    plot_loss_vs_sample_size(grid_sample_nums, grid_flux,
                             active_sample_flux_nums, active_flux)
    print("Plotting lags")
    plot_loss_vs_sample_size(grid_sample_nums, grid_lags,
                             active_sample_lags_nums, active_lags)
    
def energy_plots(dataset,scaler,model,egrid,fname,folname):
    flux_true = []
    flux_model = []
    spins = []
    masses = []
    incs = []
    rins = []
    routs = []
    index_start = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    indexes = [int(i*len(egrid)) for i in index_start]
    values = egrid[indexes]
    for batch, (D,P,M) in enumerate(dataset):
        M = np.squeeze(M)
        D = np.squeeze(D)
        #retrieve relevant data and parameters
        spin, mass, inc, rin, rout = (P[0][0].item(),10**P[0][1].item(),
                                      P[0][2].item(),P[0][3].item(),
                                      P[0][4].item())
        spins.append(spin)
        masses.append(mass)
        incs.append(inc)
        rins.append(rin)
        routs.append(rout)
        D[M==0] = 1e-38
        da = np.squeeze(D)
        
        #generate neural network prediction and rescale to linear space
        pred = model(P).detach().numpy()
        pred = 10**np.squeeze(inverse(scaler,pred))
        pred[M==0] = 1e-38
        
        flux_true.append(pred[indexes].tolist())
        flux_model.append(da[indexes].tolist())
    
    flux_true = np.asarray(flux_true)
    flux_model = np.asarray(flux_model)
    
    parameters = [spins,masses,incs,rins,routs]
    parameter_names = ["spins","masses","incs","rins","routs"]
    
    for i,energy in enumerate(values):
        data_true = flux_true[:,i]
        data_model = flux_model[:,i]
        for parameter,basename in zip(parameters,parameter_names):
            plot_resids_vs_energy(data_true, data_model, parameter, basename, energy, fname, folname)


def plot_resids_vs_energy(data_true,data_model,base,basename,energy,fname,
                          folname,scale = "linear"):
    fig, axs = plt.subplots(2,1,sharex=True)
    axs[0].scatter(base,data_true,c="blue",label="NN model",s=0.5)
    axs[0].scatter(base,data_model,c="r",label="Truth",s=0.5)
    axs[0].legend()
    axs[0].set_ylabel("Flux")
    axs[1].scatter(base,np.abs(data_true-data_model),s=0.5)
    axs[1].set_ylabel("Residuals")
    axs[1].set_xlabel(basename)
    plt.xscale(scale)
    plt.yscale("log")
    plt.title(f"Residuals as dependent on {basename} at {round(energy,2)}keV")
    plt.savefig(f"loss/{folname}/{fname}_{basename}_{round(energy,2)}.png")
    plt.close()

def plot_loss_vs_sample_size(grid_sample_nums, grid, active_sample_nums, active):
    
    fig , axs = plt.subplots(1,2,sharey=True, sharex=True, figsize=(12,9))
    
    for (x,y,z) in zip(grid_sample_nums,grid.low,grid.high):
        axs[0].scatter([x]*len(y),y, s=1, color="orange", zorder = 1, marker = "x",
                       alpha = 0.5)
        axs[0].scatter([x]*len(z),z, s=1, color="orange", zorder = 1, marker = "x",
                       alpha = 0.5)
        
    axs[0].fill_between(grid_sample_nums, grid.q_95, 
                     grid.q_5, alpha = 0.25,color = "orange",
                     zorder=3)
    axs[0].fill_between(grid_sample_nums, grid.q_75, 
                     grid.q_25, alpha = 0.5,color = "orange",
                     zorder=4)
    axs[0].fill_between(grid_sample_nums, grid.q_99, 
                     grid.q_1, alpha = 0.5,color = "orange",
                     zorder=2)
    axs[0].plot(grid_sample_nums,grid.median,label="Grid",color = "orange",
             zorder=5)
    
    for (x,y,z) in zip(active_sample_nums,active.low,active.high):
        axs[1].scatter([x]*len(y),y, s=1, color="blue", zorder = 2, marker = "x",
                       alpha = 0.5)
        axs[1].scatter([x]*len(z),z, s=1, color="blue", zorder = 2, marker = "x",
                       alpha = 0.5)
        
    axs[1].fill_between(active_sample_nums, active.q_95, 
                     active.q_5, alpha = 0.25,color = "blue",
                     zorder=3)
    axs[1].fill_between(active_sample_nums, active.q_75, 
                     active.q_25, alpha = 0.5,color = "blue",
                     zorder=4)
    axs[1].fill_between(active_sample_nums, active.q_99, 
                     active.q_1, alpha = 0.5,color = "blue",
                     zorder=2)
    axs[1].plot(active_sample_nums,active.median,label="Active learning",
             color = "blue",zorder=5)
    
    axs[0].axhline(y=1e-2, ls = "--",label="1% error",zorder=6,color="green")
    axs[1].axhline(y=1e-2, ls = "--",zorder=6,color="green")
    
    plt.yscale("log")
    plt.xscale("log")
    
    plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1])
    
    fig.supxlabel("Number of samples used in training")
    fig.supylabel("Residuals")
    lines = []
    labels = []
      
    for ax in fig.axes:
        Line, Label = ax.get_legend_handles_labels()
        # print(Label)
        lines.extend(Line)
        labels.extend(Label)
        
    fig.legend(lines, labels, loc='upper right')
    fig.tight_layout()
    plt.savefig("loss/loss_by_sample_size.png")
    plt.close()
    
def main():
    wrk_dir = os.getcwd()
    
    set_envir_vars(wrk_dir)
    
    egrid = retrieve_egrid(wrk_dir)
    lags_egrid = np.logspace(np.log10(0.5),np.log10(11),num=26)
    
    active_v_grid(wrk_dir,egrid, lags_egrid)
    
if __name__ == "__main__":
    main()
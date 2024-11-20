"""
This program serves to visualise a neural network's outputs vs the true values.
"""
from sherpa.astro.io import read_arf

import torch
import os

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
import matplotlib

import numpy as np

import pandas as pd
from tqdm import tqdm

import network
from dataStructures import PCADataset
from training import PCALoss
#from generator import new_set, lhc_AGN
from sherpa.astro import xspec
            
def inverse(scaler,data):
    """
    Transforms normalized data back to original form

    Parameters
    ----------
    scaler : scipy scaler
        scaler object that has been previously fitted on data.
    data : ndarray or other list-like object
        normalized data to transform.

    Returns
    -------
    scaled_data : ndarray
        rescaled data.

    """
    scaled_data = scaler.inverse_transform(data)
    return scaled_data

def returnContinuous(df):
    return df.data

def set_envir_vars(wrk_dir):
    """
    Sets environment variables for rtdist plotting

    Parameters
    ----------
    wrk_dir : string
        location of working directory.
    """
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

def residuals_dataframe(residuals,names):
    d = {"Residuals":[],"Sample Size":[]}
    for i,name in enumerate(names):
        print(name)
        for point in residuals[i].flatten():
            d["Residuals"].append(point)
            d["Sample Size"].append(name) 
    print("# of residuals:"+str(len(d["Residuals"])))
    print("# of labels:"+str(len(d["Sample Size"])))
    df = pd.DataFrame(data = d)
    return df

def heatmap(df, index, ticks, ticklabels, fname, mode):
    zlabel = "Fractional difference between NN model and rtdist"
    Z_center = -2.5
    #colormap
    top = cm.get_cmap('autumn', 128)
    bottom = cm.get_cmap('winter', 128)
    middle = cm.get_cmap('summer',128)

    newcolors = np.vstack((bottom(np.linspace(0, 1/2, 128)),
                           middle(np.linspace(0, 1/2, 128)),
                        top(np.linspace(2/3, 1, 128))))
    newcmp = ListedColormap(newcolors, name='summer_winter_autumn')
    
    residuals = df["residuals"].apply(returnContinuous)
    resids = []
    for item in residuals:
        resids.append(item)
    resids = np.asarray(resids)
    
    fig = plt.figure(figsize=(10,10))
    norm = colors.LogNorm(vmin = 10**(Z_center-1.5), vmax = 10**(Z_center+1.5))
    ax = plt.pcolormesh(resids, cmap=newcmp, norm=norm)
    c_ticks = [10**(Z_center-1.5), 10**(Z_center-1), 10**(Z_center-0.5), 10**(Z_center),
                10**(Z_center+0.5),10**(Z_center+1),10**(Z_center+1.5)]
    cbar = plt.colorbar(ticks=c_ticks, format='%.0e', norm=norm)
    plt.yticks(ticks,labels=ticklabels)
    if mode == "flux":
        plt.xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
    else:
        egrid = np.logspace(np.log10(0.5),np.log10(11),num=26)[:-1]
        tick_index = np.arange(0,25,24/4).astype(int)
        plt.xticks(tick_index, labels=np.round(egrid[tick_index],2))
    plt.xlabel("Energy in keV")
    plt.ylabel(index)
    
    cbar.set_label(zlabel, rotation=270, labelpad=15)
    fig.tight_layout()
    matplotlib.rcParams.update({'font.size': 16})
    plt.savefig(f"heatmaps/{fname}_{index}.pdf")
    plt.close()
    return

def loss_plots(loss_name):
    #plotting of training and validation loss over time
    PCA_train_loss = np.loadtxt(f"loss/{loss_name}_tr_loss.txt")
    PCA_val_loss = np.loadtxt(f"loss/{loss_name}_te_loss.txt")
    
    epochs = np.arange(1,len(PCA_train_loss)+1)
    plt.plot(epochs,PCA_train_loss,label = "Training loss", c = "blue",
             ls = "-")
    plt.plot(epochs,PCA_val_loss,label = "Validation loss", c = "orange",
             ls = "--")
    plt.yscale("log")
    #plt.xscale("log")
    plt.xlabel("Training epochs")
    plt.ylabel("Loss")
    #plt.title(f"Loss by epoch for {loss_name}")
    plt.legend()
    plt.tight_layout()
    plt.ylim(top = 1e1)
    plt.savefig(f"loss/loss_{loss_name}.pdf")
    plt.close()
    
    plt.plot(PCA_train_loss/PCA_val_loss,c="b",ls="-")
    plt.axhline(1,c="orange",ls="--")
    plt.xlabel("Epoch")
    plt.ylabel("Ratio between training loss and validation loss")
    plt.savefig("loss/loss_ratio.pdf")
    plt.close()

def PCA_plot(test_data,D,labels,name,pars_list,e_ticks):
    pars = np.asarray(test_data.pars)
    pca = test_data.pca
    spectra_scaler = test_data.spec_scaler
    comp_scaler = test_data.PCA_scaler
    
    scaled = spectra_scaler.transform(np.log10(D))
    pca_comps = pca.transform(scaled)
    scaled_pca_comps = comp_scaler.fit_transform(pca_comps)
    recon_D = 10**spectra_scaler.inverse_transform(pca.inverse_transform(pca.transform(scaled)))
    
    recon_resid = np.abs((D-recon_D))
    recon_perc = np.abs((D-recon_D)/D)
    recon_perc[D==1e-11] = np.nan
    
    print(f"Maximum PCA residual is {recon_perc[~np.isnan(recon_perc)].max()*100}%")
    print(f"Average PCA residual is {recon_perc[~np.isnan(recon_perc)].mean()*100}%")
    percent = (recon_perc[(recon_perc<0.01)&~np.isnan(recon_perc)].size/recon_perc[~np.isnan(recon_perc)].size)*100
    print(f"{percent}% of PCA residuals are below 1%")
    """
    plt.plot(np.mean(recon_resid,axis=0))
    plt.xlabel("Energy channel")
    plt.ylabel("Mean absolute error")
    plt.title("Absolute error of PCA reconstruction error")
    plt.yscale("log")
    plt.savefig(f"samples/{name}_PCA_mean_error.pdf")
    plt.close()
    plt.plot(np.mean(recon_perc,axis=0))
    plt.xlabel("Energy channel")
    plt.ylabel("Mean fractional error")
    plt.title("Fractional error of PCA reconstruction error")
    plt.yscale("log")
    plt.savefig(f"samples/{name}_PCA_mean_perc.pdf")
    plt.close()
    """
    for i in range(len(pars_list)):
        print(f"Creating plot for {labels[i]}")
        sort_ind = np.argsort(test_data.pars[:,i])
        sort_par = test_data.pars[sort_ind,i]
        percents = np.array([0,0.25,0.5,0.75,0.99])
        ticks = (len(sort_par)*percents).astype(int)
        tick_labels = np.asarray(np.round(sort_par[(len(sort_par)*percents).astype(int)],1)).astype(str)
        resids = recon_perc[sort_ind]
        zlabel = "Fractional difference between RTFAST and RTDIST"
        Z_center = -2.5
        #colormap
        top = cm.get_cmap('autumn', 128)
        bottom = cm.get_cmap('winter', 128)
        middle = cm.get_cmap('summer',128)

        newcolors = np.vstack((bottom(np.linspace(0, 1/2, 128)),
                               middle(np.linspace(0, 1/2, 128)),
                            top(np.linspace(2/3, 1, 128))))
        newcmp = ListedColormap(newcolors, name='summer_winter_autumn')
        newcmp.set_over('black')

        fig = plt.figure(figsize=(10,10))
        norm = colors.LogNorm(vmin = 10**(Z_center-1.5), vmax = 10**(Z_center+1.5))
        ax = plt.pcolormesh(resids, cmap=newcmp, norm=norm)
        c_ticks = [10**(Z_center-1.5), 10**(Z_center-1), 10**(Z_center-0.5), 10**(Z_center),
                    10**(Z_center+0.5),10**(Z_center+1),10**(Z_center+1.5)]
        cbar = plt.colorbar(ticks=c_ticks, format='%.0e', norm=norm, extend='max')
        plt.yticks(ticks,labels=tick_labels)
        plt.xticks(np.arange(0,recon_resid.shape[1],recon_resid.shape[1]/5), 
                   labels=e_ticks)
        plt.xlabel("Energy in keV")
        plt.ylabel(labels[i])
        
        cbar.set_label(zlabel, rotation=270, labelpad=15)
        fig.tight_layout()
        matplotlib.rcParams.update({'font.size': 16})
        plt.savefig(f"heatmaps/PCA_recon_err_{labels[i]}.png")
        plt.close()
    
    for i in range(pars.shape[1]):
        fig, axs = plt.subplots(5,5,
                                sharex=True,
                                sharey=True,
                                figsize=(20,20))
        norm = colors.Normalize(vmin=np.min(pars[:,i]),vmax=np.max(pars[:,i]))
        cmap = plt.get_cmap("plasma")
        for j in range(5):
            axs[4,j].set_xlabel(f"PCA {j}")
            for k in range(5):
                axs[j,k].scatter(scaled_pca_comps[:,j],scaled_pca_comps[:,k],
                                 c=pars[:,i],cmap=cmap,norm=norm)
                if j == 0:
                    axs[k,0].set_ylabel(f"PCA {k}")
                if k > j:
                    fig.delaxes(axs[j,k])
        
        sm =  ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[:,4],format='%.2e')
        fig.suptitle(labels[i])
        matplotlib.rcParams.update({'font.size': 16})
        plt.savefig(f"samples/corner/{labels[i]}_corner.png")
        plt.close()

def run_plot(wrk_dir,name,plot_pca = False,plot_loss = False,
             plot_heatmaps=False,plot_samples=False,
             num_models=3):
    """
    Function dedicated to plotting PCA neural network model outputs. Only plots
    things necessary for PCA model diagnosis

    Parameters
    ----------
    wrk_dir : string
        current working directory location.

    Returns
    -------
    None.

    """
    
    if plot_loss == True:
        for i in range(num_models):
            loss_plots(f"{i}_20_pars_flux")
    
    #plotting of emulator vs test data performance
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,12,13,23]
    
    labels = ["height","a","inc","rin","rout","z","Gamma","distance","Afe",
              "logNe","kte","boost","mass","honr","b1","b2","Anorm"]
    
    test_data = PCADataset("data/locations/loc_flux_text.csv",
                               pars_list,negatives,logged,scale_bool = False,
                               PCA_loc="scalers/PCA_spec.bin",
                               comp_loc="scalers/comp_spec.bin",
                               spec_scal_loc="scalers/spec_spec.bin")
    
    arf_name = wrk_dir+"/ResponseFiles/PN.arf"
    arf = read_arf(arf_name)
    egrid_lo,egrid_hi = arf.energ_lo[arf.energ_lo>0.1],arf.energ_hi[arf.energ_lo>0.1]
    
    emid = (egrid_lo+egrid_hi)/2
    
    bin_width = egrid_hi-egrid_lo
    
    percents = np.array([0,0.25,0.5,0.75,0.99])
    e_ticks = np.round(emid[(len(emid)*percents).astype(int)],1).astype(str)
    
    data = []
    for file in tqdm(test_data.locations[:1000]):
        data.append(np.loadtxt(file).reshape(1, -1))
    D = np.concatenate(data,axis=0)
    D[D<1e-11] = 1e-11
    
    if plot_pca == True:
        PCA_plot(test_data,D,labels,name,pars_list,e_ticks)
        
    if "active" in name:
        model = network.DynamicDropoutNetwork(20,test_data.pca.n_components,
                                              8,256,"GELU")
    else:
        model = network.RtdistSpec_ensemble(num_models=num_models)
        single_model = network.DynamicNetwork(17,200,12,256)
        #model = network.DynamicNetwork(17,test_data.pca.n_components,12,256)
        single_model.load_state_dict(torch.load("models/0_ensemble.pth",
                                                map_location=torch.device("cpu")))
    
    model.eval()
    
    test_pred = model(test_data.pars[:1000]).detach().numpy()
    
    loss_fn = PCALoss(test_data.pca.explained_variance_ratio_, "cpu")
    
    test_loss = loss_calc(np.asarray(test_data.data), test_pred, 
                          test_data.pca.explained_variance_ratio_)
    print(f"PCA loss reports: {loss_fn(model(test_data.pars[:1000]),test_data.data)}")
    
    print(test_loss.shape)
    print(f"Highest test loss is {test_loss.max()}")
    print(f"Average test loss is {test_loss.mean()}")
    plt.hist(test_loss,bins=50)
    plt.xlabel("Loss")
    plt.savefig(f"loss/test_dists_{name}.pdf")
    plt.close()
    
    test_pred = reconstruct_emulator(test_data, model)
    
    single_pred = reconstruct_emulator(test_data,single_model)
    single_residuals_signed = (single_pred-D)/D
        
    residuals_signed = (test_pred-D)/D
    residuals_signed[(D==1e-11)] = np.nan
    single_residuals_signed[(D==1e-11)] = np.nan
    
    fig, axs = plt.subplots(1,2,sharey=True,figsize=(10,5))
    axs[0].hist(single_residuals_signed[np.abs(single_residuals_signed)<0.2]*100,bins=80,density=True)
    fig.supxlabel("Percentage residual")
    fig.supylabel("Probability density")
    axs[0].set_xlim(-20,20)
    axs[0].axvline(0,ls="--",c="black")
    axs[0].axvline(-3,ls="--",c="red")
    axs[0].axvline(3,ls="--",c="red")
    axs[0].set_title("Single NN")
    axs[1].hist(residuals_signed[np.abs(residuals_signed)<0.2]*100,bins=80,density=True)
    axs[1].set_xlim(-20,20)
    axs[1].axvline(0,ls="--",c="black")
    axs[1].axvline(-3,ls="--",c="red")
    axs[1].axvline(3,ls="--",c="red")
    axs[1].set_title("Ensemble")
    plt.savefig(f"loss/resids_compare_{name}.pdf")
    plt.close()
    
    median_res = np.median(residuals_signed,axis=0)*100
    res_25 = np.quantile(residuals_signed,0.25,axis=0)*100
    res_75 = np.quantile(residuals_signed,0.75,axis=0)*100
    res_95 = np.quantile(residuals_signed,0.95,axis=0)*100
    res_05 = np.quantile(residuals_signed,0.05,axis=0)*100
    res_99 = np.quantile(residuals_signed,0.99,axis=0)*100
    res_01 = np.quantile(residuals_signed,0.01,axis=0)*100
    
    plt.fill_between(emid, res_01, res_99,color="b",alpha=0.2,label="99%")
    plt.fill_between(emid, res_05, res_95,color="b",alpha=0.25,label="95%")
    plt.fill_between(emid, res_25, res_75,color="b",alpha=0.5,label="50%")
    plt.plot(emid,median_res,c="b")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Percentage residuals")
    plt.xscale("log")
    plt.legend()
    plt.savefig(f"loss/energ_resid_{name}.pdf")
    plt.close()
    
    sign_res = ((test_pred)-D)/D
    sign_res[(D==1e-11)] = np.nan
    
    mean_sign_res = np.mean(sign_res,axis=1)
    
    residuals = np.abs(((test_pred)-D)/D)
    residuals[(D==1e-11)] = np.nan
    print(f"Maximum residual is {residuals[~np.isnan(residuals)].max()*100}%")
    print(f"Average residual is {residuals[~np.isnan(residuals)].mean()*100}%")
    print(f"Median residual is {np.median(residuals[~np.isnan(residuals)])*100}%")
    percent = (residuals[(residuals<0.03)&~np.isnan(residuals)].size/residuals[~np.isnan(residuals)].size)*100
    print(f"{percent}% of residuals are below 3%")
    
    single_residuals = np.abs(single_residuals_signed)
    print(f"Maximum residual is {single_residuals[~np.isnan(single_residuals)].max()*100}%")
    print(f"Average residual is {single_residuals[~np.isnan(single_residuals)].mean()*100}%")
    print(f"Median residual is {np.median(single_residuals[~np.isnan(single_residuals)])*100}%")
    percent = (single_residuals[(single_residuals<0.03)&~np.isnan(single_residuals)].size/single_residuals[~np.isnan(single_residuals)].size)*100
    print(f"{percent}% of residuals are below 3%")
    
    plt.plot(emid,np.mean(residuals,axis=0))
    plt.xlabel("Energy (keV)")
    plt.ylabel("Mean percentage residual")
    plt.title("Mean fractional error of emulator reconstruction")
    plt.yscale("log")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(f"samples/{name}_mean_errors.pdf")
    plt.close()
    
    sort_ind = np.argsort(test_data.pars[:,0])
    sort_par = test_data.pars[sort_ind,0]
    sorted_par = test_data.pars[sort_ind]
    
    resids = residuals[sort_ind]
    resids = np.mean(resids,axis=1)
    resids_2 = np.max(residuals[sort_ind],axis=1)
    print(sorted_par[np.argmin(resids_2)])
    
    plt.hist(resids,bins=100)
    plt.xlim(0,0.1)
    plt.xlabel("Fractional error")
    plt.ylabel("Occurences")
    plt.savefig("loss/hist_frac_mean.png")
    plt.close()
    
    plt.plot(np.logspace(np.min(np.asarray(sort_par)),2,1000),
             np.linspace(0,1,1000),label = "exactly average error")
    plt.plot(10**sort_par,np.cumsum(resids)/np.sum(resids),label="Test set")
    plt.xlabel("Cornal height")
    plt.ylabel("Cumulative error")
    plt.xscale("log")
    plt.legend()
    plt.savefig("loss/cum_frac_error.png")
    plt.close()
    
    plt.plot(10**sort_par,resids)
    plt.xlabel("Cornal height")
    plt.ylabel("Fractional error")
    plt.savefig("loss/error_h.png")
    plt.close()
    
    
    if plot_heatmaps == True:
        for i in range(len(pars_list)):
            print(f"Creating plot for {labels[i]}")
            sort_ind = np.argsort(test_data.pars[:,i])
            sort_par = test_data.pars[sort_ind,i]
            percents = np.array([0,0.25,0.5,0.75,0.99])
            ticks = (len(sort_par)*percents).astype(int)
            tick_labels = np.asarray(np.round(sort_par[(len(sort_par)*percents).astype(int)],1)).astype(str)
            resids = residuals[sort_ind]
            zlabel = "Fractional difference between RTFAST and RTDIST"
            Z_center = np.log10(0.03)
            #colormap
            top = cm.get_cmap('autumn', 256)
            middle = cm.get_cmap('summer',256)
            bottom = cm.get_cmap('winter', 256)
    
            newcolors = np.vstack((middle(np.linspace(0, 1/2, 192)),
                                   top(np.linspace(3/4, 1, 64))))
            newcmp = ListedColormap(newcolors, name='summer_autumn')
            newcmp.set_over('black')
            
            fig = plt.figure(figsize=(10,10))
            norm = colors.LogNorm(vmin = 10**(-3), vmax = 10**(-1))
            ax = plt.pcolormesh(resids, cmap=newcmp, norm=norm)
            c_ticks = [10**(-3), 10**(-2.5), 10**(-2), 10**(-1.5),
                        10**(-1)]
            cbar = plt.colorbar(ticks=c_ticks, format='%.0e', norm=norm, extend='max')
            plt.yticks(ticks,labels=tick_labels)
            plt.xticks(np.arange(0,residuals.shape[1],residuals.shape[1]/5), 
                       labels=e_ticks)
            plt.xlabel("Energy in keV")
            plt.ylabel(labels[i])
            
            cbar.set_label(zlabel, rotation=270, labelpad=15)
            fig.tight_layout()
            matplotlib.rcParams.update({'font.size': 16})
            plt.savefig(f"heatmaps/{name}_{labels[i]}.png")
            plt.close()
        """
        for i in range(len(pars_list)):
            print(f"Creating unsigned plot for {labels[i]}")
            sort_ind = np.argsort(test_data.pars[:,i])
            sort_par = test_data.pars[sort_ind,i]
            percents = np.array([0,0.25,0.5,0.75,0.99])
            ticks = (len(sort_par)*percents).astype(int)
            tick_labels = np.asarray(np.round(sort_par[(len(sort_par)*percents).astype(int)],2)).astype(str)
            resids = sign_res[sort_ind]
            zlabel = "Fractional difference between RTFAST and RTDIST"
            #colormap
            cmap = cm.get_cmap('autumn', 128)
            bottom = cm.get_cmap('winter', 128)
            middle = cm.get_cmap('summer',128)
    
            newcolors = np.vstack((bottom(np.linspace(0, 1/2, 128)),
                                   middle(np.linspace(0, 1/2, 128))))
            newcmp = ListedColormap(newcolors, name='summer_winter_autumn')
            newcmp.set_over('black')
            
            fig = plt.figure(figsize=(10,10))
            norm = colors.SymLogNorm(vmin = 0.1, 
                                     vmax = -0.1,base=10,
                                     linthresh=0.01)
            ax = plt.pcolormesh(resids, cmap=newcmp, norm=norm)
            cbar = plt.colorbar(format='%.0e', norm=norm, extend='max')
            plt.yticks(ticks,labels=tick_labels)
            plt.xticks(np.arange(0,residuals.shape[1],residuals.shape[1]/5), 
                       labels=e_ticks)
            plt.xlabel("Energy in keV")
            plt.ylabel(labels[i])
            
            cbar.set_label(zlabel, rotation=270, labelpad=15)
            fig.tight_layout()
            matplotlib.rcParams.update({'font.size': 16})
            plt.savefig(f"heatmaps/{name}_{labels[i]}_posneg.pdf")
            plt.close()
        """
    if plot_samples == True:
        print("Plotting samples")
        data = []
        for file in test_data.locations:
            data.append(np.loadtxt(file).reshape(1, -1))
        data = np.concatenate(data,axis=0)
        data[data<test_data.threshold] = test_data.threshold
        
        i = 0
        rtdists = []
        rtfasts = []
        for pred, D in zip(test_pred, data):
            fig, axs = plt.subplots(3,sharex=True,figsize=(10,10))
            axs[0].plot(emid,pred/bin_width,label="RTFAST")
            axs[0].plot(emid,D/bin_width,label="RTDIST", ls = "--")
            axs[0].set_ylabel("Flux (photons/cm^2/s/keV)")
            axs[0].legend()
            axs[0].set_yscale("log")
            axs[0].set_xscale("log")
            axs[1].plot(emid,(emid**2)*pred/bin_width,label="RTFAST")
            axs[1].plot(emid,(emid**2)*D/bin_width,label="RTDIST", ls = "--")
            axs[1].set_ylabel("Flux (keV^2/cm^2/s/KeV)")
            axs[1].legend()
            axs[1].set_yscale("log")
            axs[1].set_xscale("log")
            axs[2].plot(emid,(pred-D)/D)
            axs[2].fill_between(emid,-0.03,0.03,color="grey",alpha=0.1)
            axs[2].set_ylabel("(RTFAST-RTDIST)/RTDIST")
            fig.supxlabel("Energy (keV)")
            #fig.suptitle("Comparison of PCA emulator output vs expected")
            plt.tight_layout()
            plt.savefig(f"samples/{name}_energy_compare_{i}.pdf")
            plt.close()
            rtdists.append(D/bin_width)
            rtfasts.append(pred/bin_width)
            i += 1
            if i > 10:
                break
        rtdists = np.asarray(rtdists)
        rtfasts = np.asarray(rtfasts)
        np.savetxt("samples/rtdist_spectra.txt", rtdists)
        np.savetxt("samples/rtfast_spectra.txt", rtfasts)
        np.savetxt("samples/egrid.txt",emid)

def reconstruct_emulator(dataset,model):
    pred = model(dataset.pars).detach().numpy()
    pred = dataset.PCA_scaler.inverse_transform(pred)
    pred = dataset.pca.inverse_transform(pred)
    pred = dataset.spec_scaler.inverse_transform(pred)
    pred = 10**pred
    return pred

def loss_calc(data,model,variances):
    log_vars_ratios = np.log10(variances/np.min(variances))+1
    loss = np.mean(((model-data)**2)*log_vars_ratios,axis=1)
    return loss

def main():
    wrk_dir = os.getcwd()
    """
    arf_name = wrk_dir+"/ResponseFiles/PN.arf"
    arf = read_arf(arf_name)
    egrid_lo,egrid_hi = arf.energ_lo[arf.energ_lo>0.1],arf.energ_hi[arf.energ_lo>0.1]
    range_AGN = np.asarray(lhc_AGN())

    pars_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,12,13]
    
    new_set(range_AGN, pars_list, negatives, logged, egrid_lo, egrid_hi)
    """
    num_models = 7
    run_plot(wrk_dir,f"ensemble_{num_models}",plot_pca=False,plot_loss=False,
             plot_heatmaps=False,plot_samples=True,
             num_models=num_models)
    
if __name__ == "__main__":
    main()
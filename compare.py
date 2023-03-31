"""
This program serves to visualise a neural network's outputs vs the true values.
"""
import torch
import os

from sherpa.astro.ui import unpack_rmf
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from joblib import load

import pandas as pd
import seaborn as sns

import network
from main import CustomData

class LoadCustomData(CustomData):
    def __init__(self,pars,data,scaler):
        super().__init__(pars,data,scaler)
        self.par_list = pars
        self.data = data

def inverse(scaler,data):
    scaled_data = scaler.inverse_transform(data)
    return scaled_data

def residual_plots(egrid,pred,da,spin,mass,fname,log = False):
    fig, axs = plt.subplots(2,1,sharex=True)
    axs[0].plot(egrid,pred,c="blue",label="NN model")
    axs[0].plot(egrid,da,c="r",label="Truth",lw=1.)
    axs[0].legend()
    axs[0].text(0.3,0.5,f"Spin, Mass: {spin:>2f} {mass:>2f}",
                transform=axs[0].transAxes)
    if log == True:
        axs[0].set_ylabel("Log(Flux)")
    else:
        axs[0].set_ylabel("Flux")
    axs[1].scatter(egrid,(da-pred)/da,s=0.5)
    axs[1].set_ylabel("Residuals")
    axs[1].set_xlabel("Energy in keV")
    axs[1].set_ylim(residuals_ylim)
    plt.savefig(f"samples/{fname}.png")
    plt.close()
    

wrk_dir = os.getcwd()
scaler = StandardScaler()
scaler = load('std_scaler.bin')

tr_loss_arr = np.loadtxt("tr_loss.txt")
te_loss_arr = np.loadtxt("te_loss.txt")

plt.plot(tr_loss_arr,label="training loss")
plt.plot(te_loss_arr,label="testing loss")  
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.savefig("loss_plot.png")
plt.close()

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
rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
rmf = unpack_rmf(rmf_name)
egrid = rmf.e_min


model = network.NeuralNetwork(len(egrid))
model.load_state_dict(torch.load("best_model.pth"))


with open("data.txt","r") as f1:
    data = np.loadtxt(f1)

f1.close()

with open("pars.txt","r") as f2:
    pars = np.loadtxt(f2)

f2.close()

batch_size = 1
pars = pars[:,[1,13]]
pars = torch.tensor(pars)
data = torch.tensor(data)

data = LoadCustomData(pars,data,scaler)
testing_dataloader = DataLoader(data,batch_size = batch_size)

egrid = rmf.e_min

residuals_ylim = (-0.2,0.2)

for batch, (D,P) in enumerate(testing_dataloader):
    #retrieve relevant data and parameters
    spin, mass = P[0][0].item(),P[0][1].item()
    da = torch.squeeze(D)
    
    #generate neural network prediction and rescale to linear space
    pred = model(P).detach().numpy()
    pred = 10**np.squeeze(inverse(scaler,pred))
    
    fname = "{batch}_res"
    
    residual_plots(egrid, pred, da, spin, mass, fname)
    
    pred = model(P).detach().numpy()
    pred = np.squeeze(inverse(scaler,pred))
    
    da_log = np.log10(da)
    
    fname = f"{batch}_res_log"

    residual_plots(egrid, pred, da_log, spin, mass, fname)
    
    pred = model(P).detach().numpy()
    pred = np.squeeze(pred)
    
    fname = f"{batch}_res_log"
    
    da_log_scal = scaler.transform(da_log.reshape(1, -1)).flatten()

    residual_plots(egrid, pred, da_log_scal, spin, mass, fname)
    
    if batch > 30:
        break

class Residual():
    
    def __init__(self,residuals):
        self.data = residuals


print("Finished creating random model comparison")
mass, spin = [], []
residuals = []
for batch, (D,P) in enumerate(testing_dataloader):
    spin.append(P[0][0].item())
    mass.append(P[0][1].item())
    pred = model(P).detach().numpy()
    pred = 10**(inverse(scaler,pred))
    resid = (da-pred)/da
    residuals.append(np.absolute(np.asarray(resid)))
    
residuals = np.asarray(residuals)
obj_residuals = []
for res in residuals:
    obj_residuals.append(Residual(res))

dataframe = pd.DataFrame({"Spin":spin,"Mass":mass,"Residuals":obj_residuals})
dataframe.sort_values(by="Spin",inplace=True,ignore_index=True)

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

print("Building heatmaps")

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

cmap_flat = sns.color_palette("hls", 2)

fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(mass_res,cmap="vlag", vmin = 0, vmax = 0.2, center = 0.01)
ax.set_yticks(mass_tick,labels=mass_ticklabel)
ax.set_xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
ax.set_xlabel("Energy in keV")
ax.set_ylabel("Mass")
plt.savefig("heatmaps/mass_hm.png")
plt.close()
print("Continuous mass hm plotted")

fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(mass_res_flat,cmap=cmap_flat)
ax.set_yticks(mass_tick,labels=mass_ticklabel)
ax.set_xticks(np.arange(0,4096,4096/4), labels=np.arange(0,20,5))
ax.set_xlabel("Energy in keV")
ax.set_ylabel("Mass")
plt.savefig("heatmaps/flat_mass_hm.png")
plt.close()
print("Flat mass hm plotted")

fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(spin_res,cmap="vlag", vmin = 0, vmax = 0.2, center = 0.01)
ax.set_yticks(spin_tick,labels=spin_ticklabel)
ax.set_xticks(np.arange(0,4096,4096/4),labels=np.arange(0,20,5))
ax.set_xlabel("Energy in keV")
ax.set_ylabel("Spin")
plt.savefig("heatmaps/spin_hm.png")
plt.close()
print("Continuous spin hm plotted")

fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(spin_res_flat,cmap=cmap_flat)
ax.set_yticks(spin_tick,labels=spin_ticklabel)
ax.set_xticks(np.arange(0,4096,4096/4),labels=np.arange(0,20,5))
ax.set_xlabel("Energy in keV")
ax.set_ylabel("Spin")
plt.savefig("heatmaps/flat_spin_hm.png")
plt.close()
print("Flat spin hm plotted")
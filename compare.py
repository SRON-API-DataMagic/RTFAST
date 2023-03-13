"""
This program serves to visualise a neural network's outputs vs the true values.
"""
import torch
import generator
import network
import os
from sherpa.astro.ui import unpack_rmf
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Dataset

class custom_data(Dataset):
    
    def __init__(self,pars,data):
        super().__init__()
        self.par_list = pars
        self.data = data
    
    def __len__(self):
        return self.par_list.shape[0]
    
    def __getitem__(self,idx):
        datum = torch.from_numpy(self.data[idx])
        parameters = torch.from_numpy(self.par_list[idx])
        return datum,parameters
    
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
rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
rmf = unpack_rmf(rmf_name)
egrid = rmf.e_min


model = network.NeuralNetwork(len(egrid))
model.load_state_dict(torch.load("best_model.pth"))
n = 10

with open("data.npy","rb") as f1:
    data = np.load(f1)

with open("pars.npy","rb") as f2:
    pars = np.load(f2)

batch_size = 1
pars = pars[:,[1,13]]
data = custom_data(pars,data)
testing_dataloader = DataLoader(data,batch_size = batch_size)

egrid = rmf.e_min[100:]

for batch, (D,P) in enumerate(testing_dataloader):
    fig, axs = plt.subplots(2,1,sharex=True)
    pred = model(P)
    pred = 10**np.squeeze(pred.detach().numpy())[100:]
    da = 10**torch.squeeze(torch.log10(D))[100:]
    axs[0].plot(egrid,pred,c="blue",label="NN model")
    axs[0].plot(egrid,da,c="r",label="Truth",lw=1.)
    axs[0].legend()
    axs[0].set_ylabel("Flux")
    axs[1].scatter(egrid,(da-pred)/da,s=0.5)
    axs[1].set_ylabel("Residuals")
    axs[1].set_xlabel("Energy in keV")
    plt.savefig(f"samples/{batch}_best_com.png")
    plt.close()
    if batch > 30:
        break

for batch, (D,P) in enumerate(testing_dataloader):
    fig, axs = plt.subplots(2,1,sharex=True)
    pred = model(P)
    pred = np.squeeze(pred.detach().numpy())[100:]
    da = torch.squeeze(torch.log10(D))[100:]
    axs[0].plot(egrid,pred,c="blue",label="NN model")
    axs[0].plot(egrid,da,c="r",label="Truth",lw=1.)
    axs[0].legend()
    axs[0].set_ylabel("Flux")
    axs[1].scatter(egrid,(da-pred)/da,s=0.5)
    axs[1].set_ylabel("Residuals")
    axs[1].set_xlabel("Energy in keV")
    plt.savefig(f"samples/{batch}_log_best_com.png")
    plt.close()
    if batch > 30:
        break

model.load_state_dict(torch.load("final_model.pth"))

for batch, (D,P) in enumerate(testing_dataloader):
    fig, axs = plt.subplots(2,1,sharex=True)
    pred = model(P)
    pred = 10**np.squeeze(pred.detach().numpy())[100:]
    da = 10**torch.squeeze(torch.log10(D))[100:]
    axs[0].plot(egrid,pred,c="blue",label="NN model")
    axs[0].plot(egrid,da,c="r",label="Truth",lw=1.)
    axs[0].legend()
    axs[0].set_ylabel("Flux")
    axs[1].scatter(egrid,(da-pred)/da,s=0.5)
    axs[1].set_ylabel("Residuals")
    axs[1].set_xlabel("Energy in keV")
    plt.savefig(f"samples/{batch}_final_com.png")
    plt.close()
    if batch > 30:
        break

for batch, (D,P) in enumerate(testing_dataloader):
    fig, axs = plt.subplots(2,1,sharex=True)
    pred = model(P)
    pred = np.squeeze(pred.detach().numpy())[100:]
    da = torch.squeeze(torch.log10(D))[100:]
    axs[0].plot(egrid,pred,c="blue",label="NN model")
    axs[0].plot(egrid,da,c="r",label="Truth",lw=1.)
    axs[0].legend()
    axs[0].set_ylabel("Flux")
    axs[1].scatter(egrid,(da-pred)/da,s=0.5)
    axs[1].set_ylabel("Residuals")
    axs[1].set_xlabel("Energy in keV")
    plt.savefig(f"samples/{batch}_log_final_com.png")
    plt.close()
    if batch > 30:
        break
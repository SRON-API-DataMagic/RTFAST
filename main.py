"""
This is the main program that trains the neural network.
"""

import generator
import network
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor

from sherpa.astro.ui import unpack_rmf
import os
import numpy as np


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

def pregenerate_models(n,egrid):
    all_data = []
    all_pars = []
    for i in range(n):
        print(f"Generating model {i+1}")
        pars = generator.pregen()
        data = generator.rtdist_flux(pars, egrid)
        all_data.append(data)
        all_pars.append(pars)
    all_data = np.asarray(all_data)
    all_pars = np.asarray(all_pars)
    with open("data.npy","wb") as f1:
        np.save(f1,all_data)
    with open("pars.npy","wb") as f2:
        np.save(f2,all_pars)
    f1.close()
    f2.close()
    return all_data,all_pars

def save_data(data,pars):
    data = np.asarray(data)
    pars = np.asarray(pars)
    with open("data.npy","ab") as f:
        np.save(f,data)
    f.close()
    with open("pars.npy","ab") as f:
        np.save(f,pars)
    f.close()
    return
    

def train(dataloader,model,optimizer,loss_fn,true_func,par_gen,egrid):
    model.train()
    batches = 10000
    batch = 0
    for D,P in dataloader:
        print(P)
        pred = ToTensor(model(P))
        loss = loss_fn(pred,D)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{batches:>5d}]")
        batch += 1
        
    return model, optimizer

def test(dataloader,model,loss_fn,true_func,par_gen,egrid):
    model.eval()
    test_loss = 0
    batches = 1000
    with torch.no_grad():
        for batch in range(batches):
            pars = par_gen()
            truth = true_func(pars)
            pred = model(pars)
            test_loss += loss_fn(pred, truth).item()
    test_loss /= batches
    print(f"Avg loss: {test_loss:>8f}")

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

data,pars = pregenerate_models(20, egrid)
test_data = custom_data(pars, data)

with open("data.npy","rb") as f1:
    data = np.load(f1)

with open("pars.npy","rb") as f2:
    pars = np.load(f2)

data = custom_data(pars,data)

batch_size = 1
training_dataloader = DataLoader(data,batch_size = batch_size)
testing_dataloader = DataLoader(test_data,batch_size = batch_size)

model = network.NeuralNetwork(pars.shape[1],len(egrid))
optimizer = 0
max_iters = 10000
i = 0

print("Beginning training")
while i < max_iters:
    print(f"Epoch {i+1} \n -----------------------")
    model, optimizer = train(training_dataloader,model,optimizer,nn.MSELoss,generator.rtdist_lags,generator.par_gen,egrid)
    test(testing_dataloader,model,nn.MSEloss,generator.rtdist_lags,generator.par_gen,egrid)
    
print("Completed training")
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

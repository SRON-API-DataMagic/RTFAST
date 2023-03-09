"""
This is the main program that trains the neural network.
"""

import generator
import network
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sherpa.astro.ui import unpack_rmf
import os
import numpy as np
import pandas as pd

def pregenerate_models(n,egrid):
    all_data = []
    all_pars = []
    for i in range(n):
        print(f"Generating model {i+1}")
        pars = generator.pregen()
        data = generator.rtdist_flux(pars, egrid)
        all_data.append(data)
        all_pars.append(pars)
        pars = np.asarray(pars)
        data = np.asarray(data)
        with open("data.npy","wb") as f1:
            np.save(f1,data)
        with open("pars.npy","wb") as f2:
            np.save(f2,pars)
    
    f1.close()
    f2.close()
    return all_data,all_pars

def save_data(data,pars):
    data = np.asarray(data)
    pars = np.asarray(pars)
    with open("data.npy","wb") as f:
        np.save(f,data)
    f.close()
    with open("pars.npy","wb") as f:
        np.save(f,pars)
    f.close()
    return
    

def train(model,optimizer,loss_fn,true_func,par_gen):
    model.train()
    batches = 10000
    for batch in range(batches):
        pars = par_gen()
        truth = true_func(pars)
        pred = ToTensor(model(pars))
        loss = loss_fn(pred,truth)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{batches:>5d}]")
    
    return model, optimizer

def test(model,loss_fn,true_func,par_gen):
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
rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
rmf = unpack_rmf(rmf_name)
egrid = rmf.e_min

data,pars = pregenerate_models(20, egrid)

model = network.NeuralNetwork(len(egrid))
optimizer = 0
max_iters = 10000
i = 0

print("Beginning training")
while i < max_iters:
    print(f"Epoch {i+1} \n -----------------------")
    train(model,optimizer,nn.MSELoss,generator.rtdist_lags,generator.par_gen)
    test(model,nn.MSEloss,generator.rtdist_lags,generator.par_gen)
    
print("Completed training")
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

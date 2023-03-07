"""
This is the main program that trains the neural network.
"""

import generator
import network
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sherpa.astro.ui import unpack_arf,unpack_rmf
import os

def train(model,optimizer,loss_fn,true_func,par_gen):
    model.train()
    batches = 1000
    for batch in range(batches):
        pars = par_gen()
        truth = true_func(pars)
        pred = model(pars)
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

model = network.NeuralNetwork()
optimizer = 0
max_iters = 10000
i = 0

print("Beginning training")
while i < max_iters:
    print(f"Epoch {i+1} \n -----------------------")
    train(model,optimizer,nn.MSEloss,generator.rtdist_lags,generator.par_gen)
    test(model,nn.MSEloss,generator.rtdist_lags,generator.par_gen)
    
print("Completed training")
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

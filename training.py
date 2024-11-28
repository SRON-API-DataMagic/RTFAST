"""
This program holds the base training and testing processes for neural network
training.
"""

import torch
from torch import nn
import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
from processing import mergeSaveData, saveLoop

def model_NaN_checker(D,P,model):
    """
    This function checks for the cause of models parameters going to zero by
    checking for infinities and NaNs in the model parameters, data and input
    parameters.

    Parameters
    ----------
    D : torch.array
        array of data.
    P : torch.array
        array of input parameters.
    model : neural network
        neural network to check.

    Returns
    -------
    None.

    """
    #Checks data for NaNs and infinities
    if torch.any(torch.isnan(D)) == True:
        print("D contains NaNs")
        print(D)
        quit()
    elif torch.any(torch.isinf(D)) == True:
        print("D contains infinities")
        print(D)
        quit()
    #Checks input parmaeters for NaNs and infinities
    if torch.any(torch.isnan(P)) == True:
        print("P contains NaNs")
        print(P)
        quit()
    elif torch.any(torch.isinf(P)) == True:
        print("D contains infinities")
        print(P)
        quit()
    #Checks model parmaeters for NaNs and infinities
    for name, param in model.named_parameters():
        if torch.any(torch.isnan(param.data)) == True:
            print(f"Param is {param.data}")
            print("Exiting program")
            quit()
        elif torch.any(torch.isinf(param.data)) == True:
            print(f"Param is {param.data}")
            print("Exiting program")
            quit()
        if torch.any(param.data > 100) == True:
            print(f"Parameters exceed 100: {param.data}")

class PCALoss(nn.Module):
    
    def __init__(self,variances,device,verbose=False,individual=False):
        super().__init__()
        log_vars_ratios = np.log10(variances/np.min(variances))+1
        self.variances = torch.tensor(log_vars_ratios)
        self.device = device
        self.individual = individual
        if verbose==True:
            print(self.variances)
    
    def forward(self,pred,target):
        return self.weightedMSELoss(torch.squeeze(pred),target)
    
    def weightedMSELoss(self,pred,target):
        loss = (pred-target)**2
        weighted_loss = torch.mul(loss,self.variances.to(self.device))
        if self.individual == False:
            return torch.mean(weighted_loss)
        else:
            return torch.mean(weighted_loss,1)

def train_flux(dataloader, model, optimizer, loss_fn, device, scheduler = None,
               epoch = 0):
    """
    

    Parameters
    ----------
    dataloader : torch.nn.utils.data.DataLoader
        provides iterable shuffled form of the training dataset.
    model : network.NeuralNetwork
        the neural network model to be trained.
    optimizer : torch.optim
        optimizer used for training the network.
    loss_fn : torch.nn loss function
        loss function used to train the network.

    Returns
    -------
    model : network.NeuralNetwork
        the neural network model to be trained.
    optimizer : Ttorch.optim
        optimizer used for training the network.
    avg_loss : float
        used as to record and determine how many iterations should be trained.

    """
    
    model.train()
    
    size = len(dataloader.dataset)
    batches = size/1024
    loss_tot = 0
    loss_arr = []
    iters = len(dataloader)
    for batch, (D,P) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(P.to(device))
        loss = loss_fn(pred,D.to(device))
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step(epoch + batch / iters)
        loss_b = loss.detach().item()
        loss_tot += loss_b
        loss_arr.append(loss_b)
        if batch % np.ceil(batches*0.1) == 0:
            current = ((batch+1)*P.shape[0])
            print(f"loss: {loss_b:>7f}  [{current:>5d}/{size:>5d}]")
    
    avg_loss = loss_tot/len(dataloader)
    print(f"Average training loss: {avg_loss:>8f}")
    return model, optimizer , avg_loss

def test_flux(dataloader, model, loss_fn, device):
    """
    

    Parameters
    ----------
    dataloader : torch.nn.utils.data.DataLoader
        provides iterable shuffled form of the testing dataset.
    model : network.NeuralNetwork
        the neural network model to be tested.
    loss_fn : torch.nn loss function
        loss function used to train the network.

    Returns
    -------
    test_loss : float
        used as to record and determine how many iterations should be trained.

    """
    model.eval()
    test_loss = 0
    batches = len(dataloader)
    
    with torch.no_grad():
        for batch, (D,P) in enumerate(dataloader):
            pred = model(P.to(device))
            test_loss += loss_fn(pred,D.to(device)).detach().item()
    test_loss /= batches
    
    print(f"Average testing loss: {test_loss:>8f}")
    return test_loss

def training_loop(model, optimizer, train, test, train_dataloader, 
                    test_dataloader, loss_fn, device, name, mode, 
                    epochs = 400, scheduler = None):
    tr_loss_arr = []
    te_loss_arr = []
    
    epoch = 0
    imp_flag = 0
    loss_best = 20
    print("Beginning training")
    
    while epoch < epochs and imp_flag < 100:
        imp_flag += 1
        #time_st = time.time()
        print(f"Epoch {epoch+1} \n -----------------------")
        model,optimizer,train_loss = train(train_dataloader,model,optimizer, 
                                           loss_fn,device,scheduler,epoch)
        val_loss = test(test_dataloader, model, loss_fn, device)
        if scheduler != None:
            scheduler.step(val_loss)
        te_loss_arr.append(val_loss)
        tr_loss_arr.append(train_loss)
        if val_loss == np.min(te_loss_arr):
            print(f"New best testing loss: {val_loss}")
            torch.save(model.state_dict(), f"models/{name}_{mode}.pth")
        if val_loss < 0.99*loss_best:
            imp_flag = 0
            loss_best = val_loss
        epoch += 1
    
    print("Completed training")
    print("Final best training loss:", np.min(tr_loss_arr))
    print("Final best testing loss:", np.min(te_loss_arr))
    torch.save(model.state_dict(), f"models/{name}_{mode}_final.pth")
    print(f"Saved PyTorch Model State to {name}_{mode}_final.pth")
    
    tr_loss_arr = np.asarray(tr_loss_arr)
    te_loss_arr = np.asarray(te_loss_arr)
    
    np.savetxt(f"loss/{name}_{mode}_te_loss.txt",te_loss_arr)
    np.savetxt(f"loss/{name}_{mode}_tr_loss.txt",tr_loss_arr)
    
    return
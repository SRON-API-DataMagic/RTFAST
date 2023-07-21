"""
This program holds the base training and testing processes for neural network
training.
"""

import torch
from torch import nn
from joblib import load
import numpy as np
import pandas as pd
from processing import mergeSaveData, saveLoop


class barredMSELoss(nn.Module):
    def __init__(self,scaler,device):
        super(barredMSELoss, self).__init__()
        self.scaler = load(f'scalers/{scaler}')
        self.device = device
        self.set_scale()
        
    def set_scale(self):
        self.min = torch.tensor(self.scaler.data_min_)
        self.max = torch.tensor(self.scaler.data_max_)
        self.scale = self.max - self.min
        
    def scaling(self,a):
        result = (a * self.scale.to(self.device)) + self.min.to(self.device)
        result = 10**result
        return result
        
    def forward(self, output, target):
        #scale to real space
        scaled_tar = self.scaling(target)
        scaled_out = self.scaling(output)
        #find desired boundaries of the original data
        data_low = 0.995*scaled_tar
        data_high = 1.005*scaled_tar
        #create mask where prediction is within boundaries
        mask = torch.where((data_low < scaled_out)&(data_high>scaled_out),0,1)
        #multiply with mask to only consider where network is out of bounds
        pred = torch.mul(output,mask)
        data = torch.mul(target,mask)
        #calculate loss
        criterion = nn.MSELoss()
        loss = criterion(pred,data)
        return loss 

class lagLoss(nn.Module):
    def __init__(self,scaler,device):
        super().__init__()
        self.scaler = load(f'scalers/{scaler}')
        self.device = device
        self.set_scale()
        self.criterion = nn.MSELoss()
        self.binary = nn.BCELoss()
    
    def set_scale(self):
        self.min = torch.tensor(self.scaler.data_min_)
        self.max = torch.tensor(self.scaler.data_max_)
        self.scale = self.max - self.min
    
    def scaling(self,a):
        result = (a * self.scale.to(self.device)) + self.min.to(self.device)
        result = 10**result
        return result
    
    def forward(self, output, target, index, index_target):
        threshold = 1e-9
        #scale to real space
        scaled_tar = self.scaling(target)
        scaled_out = self.scaling(output)
        #find desired boundaries of the original data
        data_low = 0.995*scaled_tar
        data_high = 1.005*scaled_tar
        #create mask where prediction is within boundaries
        mask = torch.where((data_low < scaled_out)&(data_high>scaled_out),0,1)
        #create mask where output is too small to measure
        small_mask = torch.where((scaled_tar > threshold)&((0<scaled_out<threshold)),0,1)
        #create overall mask
        mask = mask*small_mask
        #multiply with mask to only consider where network is out of bounds
        pred = torch.mul(output,mask)
        data = torch.mul(target,mask)
        #calculate loss
        loss = self.criterion(pred,data)
        signed_loss = self.binary(index,index_target)
        loss += signed_loss
        return loss 
    
def train_flux(dataloader,model,optimizer,loss_fn,device):
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
    loss_arr = 0
    for batch, (D,P) in enumerate(dataloader):
        pred = model(P.to(device))[:,None,:]
        loss = loss_fn(pred,D.to(device))
        optimizer.zero_grad()
        loss.backward()
        
        threshold = 0
        vanishing_grads = 0
        for p in model.parameters():
            if p.grad.norm() == threshold:
                vanishing_grads += 1
        
        if vanishing_grads != 0:
            print(f"{vanishing_grads} gradients are approaching 0")
        
        optimizer.step()
        loss_b = loss.detach().item()
        if batch % 5 == 0:
            current = (batch*P.shape[0] + 1)
            print(f"loss: {loss_b:>7f}  [{current:>5d}/{size:>5d}]")
        loss_arr += loss_b
    
    avg_loss = loss_arr/len(dataloader)
    print(f"Average training loss: {avg_loss:>8f}")
    return model, optimizer , avg_loss

def train_lags(dataloader,model,optimizer,loss_fn,device):
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
    loss_arr = 0
    for batch, (D,P) in enumerate(dataloader):
        pred = model(P.to(device))[:,None,:]
        loss = loss_fn(pred,D.to(device))
        optimizer.zero_grad()
        loss.backward()
        
        threshold = 0
        vanishing_grads = 0
        for p in model.parameters():
            if p.grad.norm() == threshold:
                vanishing_grads += 1
        
        if vanishing_grads != 0:
            print(f"{vanishing_grads} gradients are approaching 0")
        
        optimizer.step()
        loss_b = loss.detach().item()
        if batch % 5 == 0:
            current = (batch*P.shape[0] + 1)
            print(f"loss: {loss_b:>7f}  [{current:>5d}/{size:>5d}]")
        loss_arr += loss_b
    
    avg_loss = loss_arr/len(dataloader)
    print(f"Average training loss: {avg_loss:>8f}")
    return model, optimizer , avg_loss

def test_flux(dataloader,model,loss_fn,device):
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
            pred = model(P.to(device))[:,None,:]
            test_loss += loss_fn(pred, D.to(device)).detach().item()
    test_loss /= batches
    
    print(f"Average testing loss: {test_loss:>8f}")
    return test_loss

def test_lags(dataloader,model,loss_fn,device):
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
            pred = model(P.to(device))[:,None,:]
            test_loss += loss_fn(pred, D.to(device)).detach().item()
    test_loss /= batches
    
    print(f"Average testing loss: {test_loss:>8f}")
    return test_loss

def active_training_loop(model,dataloader,optimizer,loss_fn,device,
                  test_dataloader,te_loss_arr,tr_loss_arr,
                  last_sig_te,last_sig_tr, active_loop_num,
                  loop_epochs, best_model, train, test,
                  mode = "flux"):
    epoch = 0
    #set improvements counters to 0
    imp_te = 0
    imp_tr = 0
    
    print(f"Training {mode} model")
    while (imp_te < 15 or imp_tr < 15):
        print(f"Epoch {epoch+1} \n -----------------------")
        
        model, optimizer, train_loss = train(dataloader,model,
                                             optimizer,loss_fn,
                                             device)
        loss = test(test_dataloader,model,loss_fn,
                         device)
        #scheduler.step(loss)
        te_loss_arr.append(loss)
        tr_loss_arr.append(train_loss)
        
        tr_bet = (0.9*last_sig_tr) - train_loss
        te_bet = (0.9*last_sig_te) - loss
        if tr_bet > 0 and te_bet > 0:
            last_sig_tr = train_loss
            last_sig_te = loss
            imp_te = 0
            imp_tr = 0
            print(f"New sig best training loss: {train_loss}")
            print(f"New sig best testing loss: {loss}")
        elif tr_bet > 0:
            imp_tr = 0
            imp_te += 1
            last_sig_tr = train_loss
            print(f"New sig best training loss: {train_loss}")
        elif te_bet > 0:
            imp_tr += 1
            imp_te = 0
            last_sig_te = loss
            print(f"New sig best testing loss: {loss}")
        else:
            imp_te += 1
            imp_tr += 1
        if loss == np.asarray(te_loss_arr).min():
            torch.save(model.state_dict(), "models/active_{mode}_best.pth")
        
        epoch += 1
    
    if active_loop_num != 0:
        loop_epochs.append(loop_epochs[active_loop_num-1]+epoch)
    else:
        loop_epochs.append(epoch)
        
    mergeSaveData(pd.read_csv(f"data/locations/active_{mode}_test_locs.csv"), 
                  pd.read_csv(f"data/locations/active_{mode}_locs.csv"), 
                  "data/locations/",f"active_{mode}_locs.csv")
    
    temp_te = np.asarray(te_loss_arr)
    temp_tr = np.asarray(tr_loss_arr)
    temp_epochs = np.asarray(loop_epochs)
    try:
        best_model.load_state_dict(torch.load("models/active_{mode}_best.pth"))
    except:
        best_model.load_state_dict(model.state_dict())
        
    saveLoop(best_model, "data/locations/active_{mode}_locs.csv", 
             optimizer,
             temp_te, temp_tr, 
             active_loop_num, temp_epochs,
             typ=mode)
    
    return (model, best_model, optimizer, loop_epochs, 
            te_loss_arr, tr_loss_arr, last_sig_tr, last_sig_te)

def grid_training_loop(model, optimizer, train, test, 
                       train_dataloader, test_dataloader,
                       loss_fn, device,
                       size, mode):
    
    last_sig_best_tr = 1e7 #last significant best training loss (set large initially)
    last_sig_best_te = 1e7 #last significant best testing loss (set large initially)
    tr_loss_arr = []
    te_loss_arr = []
    
    epoch = 0
    imp_te = 0
    imp_tr = 0
    
    print("Beginning training")
    while epoch < 400:
        print(f"Epoch {epoch+1} \n -----------------------")
        model, optimizer, train_loss = train(train_dataloader,model,
                                             optimizer,loss_fn,device)
        loss = test(test_dataloader,model,loss_fn,device)
        te_loss_arr.append(loss)
        tr_loss_arr.append(train_loss)
        tr_bet = (0.9*last_sig_best_tr) - train_loss
        te_bet = (0.9*last_sig_best_te) - loss
        if tr_bet > 0 and te_bet > 0:
            last_sig_best_tr = train_loss
            last_sig_best_te = loss
            imp_te = 0
            imp_tr = 0
            print(f"New best training loss: {train_loss}")
            print(f"New best testing loss: {loss}")
            torch.save(model.state_dict(), f"models/grid_{size}.pth")
        elif tr_bet > 0:
            imp_tr = 0
            imp_te += 1
            last_sig_best_tr = train_loss
            print(f"New best training loss: {train_loss}")
        elif te_bet > 0:
            imp_tr += 1
            imp_te = 0
            last_sig_best_te = loss
            print(f"New best testing loss: {loss}")
            torch.save(model.state_dict(), f"models/grid_{size}.pth")
        else:
            imp_te += 1
            imp_tr += 1
        epoch += 1
    
    print("Completed training")
    print("Final best training loss:", last_sig_best_tr)
    print("Final best testing loss:", last_sig_best_te)
    torch.save(model.state_dict(), f"models/grid_{size}_{mode}_final.pth")
    print(f"Saved PyTorch Model State to grid_{size}_{mode}_final.pth")
    
    tr_loss_arr = np.asarray(tr_loss_arr)
    te_loss_arr = np.asarray(te_loss_arr)
    
    np.savetxt(f"loss/grid_{size}_{mode}_te_loss.txt",te_loss_arr)
    np.savetxt(f"loss/grid_{size}_{mode}_tr_loss.txt",tr_loss_arr)
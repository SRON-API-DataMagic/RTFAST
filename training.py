"""
This program holds the base training and testing processes for neural network
training.
"""

import torch
from torch import nn

from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

from processing import mergeSaveData, saveLoop
from math import ceil
from tqdm import tqdm
from generator import active_learning_generation

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

class FluxLoss(nn.Module):
    """
    Class of loss functon that only induces loss for values extending outside
    a given range (0.05%) of the original data
    
    -------------
    Parameters:
        scaler:
            loads a MinMaxScaler from scikitlearn that allows retrieval of
            original data
        device:
            pytorch device to load tensors into for manipulation
        min:
            minimum value of non-normalized form of data
        max:
            maximum value of non-normalized form of data
        scale:
            range of non-normalized form of data
    
    -------------
    Methods:
        __init__():
            initalizes key parameters
        
        set_scale():
            used in intializing to calculate rescaling method
        
        scaling(a):
            a is data vector to be rescaled
        
        forward(output, target):
            forward pass that returns a loss value based on mean squared error
            and masking.
    """
    def __init__(self,scaler,device):
        super().__init__()
        self.scaler = load(f'scalers/{scaler}')
        self.device = device
        self.lower_threshold = 1e-11
        self.set_scale()
        
    def set_scale(self):
        if isinstance(self.scaler, MinMaxScaler):
            self.min = torch.tensor(self.scaler.data_min_)
            self.max = torch.tensor(self.scaler.data_max_)
            self.scale = self.max - self.min
            self.scale_type = "MinMax"
        elif isinstance(self.scaler, StandardScaler):
            self.mean = torch.tensor(self.scaler.mean_)
            self.scale = torch.tensor(self.scaler.scale_)
            self.scale_type = "Standard"
        
    def scaling(self,a,testing=False):
        if self.scale_type == "MinMax":
            result = (a * self.scale.to(self.device)) + self.min.to(self.device)
            result = 10**result
            return result
        elif self.scale_type == "Standard":
            result = (a - self.mean.to(self.device))/self.scale.to(self.device)
            return 10**result
        
    def forward(self, output, target):
        #scale to real space
        scaled_tar = self.scaling(target)
        scaled_out = self.scaling(output)
        #create mask where prediction is within boundaries
        mask = torch.where(((scaled_tar<=self.lower_threshold)&
                            (scaled_out<=self.lower_threshold)),
                           0,1)
        #multiply with mask to only consider where network is out of bounds
        pred = torch.mul(output,mask)
        data = torch.mul(target,mask)
        #calculate loss
        criterion = nn.MSELoss()
        loss = criterion(pred,data)
        if torch.isnan(loss) == True:
            print("Loss has become NaN, performing checks")
            print(f"Prediction: {pred}")
            print(f"Raw output: {output}")
            print(f"Target: {target}")
            print(f"Data: {data}")
            quit()
        elif torch.isinf(loss) == True:
            print("Loss has become inf, performing checks")
            print(f"Prediction: {pred}")
            print(f"Raw output: {output}")
            print(f"Target: {target}")
            print(f"Data: {data}")
            quit()
        return loss 

class LagLoss(nn.Module):
    """
    Class of loss functon that only induces loss for values extending outside
    a given range (0.5%) of the original data. Features a threshold value that 
    all data below must be within the threshold.
    
    -------------
    Parameters:
        scaler:
            loads a MinMaxScaler from scikitlearn that allows retrieval of
            original data
        device:
            pytorch device to load tensors into for manipulation
        min:
            minimum value of non-normalized form of data
        max:
            maximum value of non-normalized form of data
        scale:
            range of non-normalized form of data
    
    -------------
    Methods:
        __init__():
            initalizes key parameters
        
        set_scale():
            used in intializing to calculate rescaling method
        
        scaling(a):
            a is data vector to be rescaled
        
        forward(output, target, index, index_target):
            forward pass that returns a loss value based on mean squared error
            and masking. Loss value is the sum of the binary criterion loss of
            getting the signed value of outputs correct and the mean squared
            error of the output, once again masked when within a certain 
            boundary.
    """
    def __init__(self,scaler,device):
        super().__init__()
        self.scaler = load(f'scalers/{scaler}')
        self.device = device
        self.set_scale()
        self.criterion = nn.MSELoss()
        self.binary = nn.BCELoss()
        self.threshold = 1e-5
    
    def set_scale(self):
        if isinstance(self.scaler, MinMaxScaler):
            self.min = torch.tensor(self.scaler.data_min_)
            self.max = torch.tensor(self.scaler.data_max_)
            self.scale = self.max - self.min
            self.scale_type = "MinMax"
        elif isinstance(self.scaler, StandardScaler):
            self.mean = torch.tensor(self.scaler.mean_)
            self.scale = torch.tensor(self.scaler.scale_)
            self.scale_type = "Standard"
        
    def scaling(self,a):
        if self.scale_type == "MinMax":
            result = (a * self.scale.to(self.device)) + self.min.to(self.device)
            result = 10**result
            return result
        elif self.scale_type == "Standard":
            result = (a - self.mean.to(self.device))/self.scale.to(self.device)
            return 10**result
    
    def forward(self, output, index, target, index_target):
        #scale to real space
        scaled_tar = self.scaling(target)
        scaled_out = self.scaling(output)
        #create mask where output is too small to measure
        mask = torch.where(((scaled_tar<=self.threshold)&
                            (scaled_out<=self.threshold)),0,1)
        #create overall mask
        #multiply with mask to only consider where network is out of bounds
        pred = torch.mul(output,mask)
        data = torch.mul(target,mask)
        #calculate loss
        loss = self.criterion(pred,data)
        signed_loss = self.binary(index,index_target)
        loss += signed_loss
        return loss 

def train_bottle(dataloader, model, optimizer, loss_fn, device, mask):
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
        P = P*mask
        optimizer.zero_grad()
        pred = model(P.to(device))
        loss = loss_fn(pred,D.to(device))
        loss.backward()
        #prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        loss_b = loss.detach().item()
        if batch % 5 == 0:
            current = ((batch+1)*P.shape[0])
            print(f"loss: {loss_b:>7f}  [{current:>5d}/{size:>5d}]")
        loss_arr += loss_b
    
    avg_loss = loss_arr/len(dataloader)
    print(f"Average training loss: {avg_loss:>8f}")
    return model, optimizer , avg_loss


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
        #prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler != None:
            scheduler.step(epoch + batch / iters)
        loss_b = loss.detach().item()
        loss_tot += loss_b
        loss_arr.append(loss_b)
        if batch % int(batches*0.1) == 0:
            current = ((batch+1)*P.shape[0])
            print(f"loss: {loss_b:>7f}  [{current:>5d}/{size:>5d}]")
    
    avg_loss = loss_tot/len(dataloader)
    med_loss = np.median(loss_arr)
    std_loss = np.std(loss_arr)
    print(f"Average training loss: {avg_loss:>8f}")
    print(f"Median training loss: {med_loss:>8f}")
    return model, optimizer , avg_loss, med_loss, std_loss

def train_lags(dataloader, model, optimizer, loss_fn, device):
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
    loss_arr = 0
    for batch, (D, I, P) in enumerate(dataloader):
        optimizer.zero_grad()
        pred, I_pred = model(P.to(device))
        pred, I_pred = pred[:,None,:], I_pred[:,None,:]
        loss = loss_fn(pred, I_pred, D.to(device), I.to(device))
        loss.backward()
        #prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_b = loss.detach().item()
        if batch % int(batches*0.1) == 0:
            current = ((batch+1)*P.shape[0])
            print(f"loss: {loss_b:>7f}  [{current:>5d}/{size:>5d}]")
        loss_arr += loss_b
        
    avg_loss = loss_arr/len(dataloader)
    print(f"Average training loss: {avg_loss:>8f}")
    return model, optimizer , avg_loss

def test_bottle(dataloader, model, loss_fn, device,mask):
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
            P = P*mask
            pred = model(P.to(device))
            test_loss += loss_fn(pred,D.to(device)).detach().item()
    test_loss /= batches
    
    print(f"Average testing loss: {test_loss:>8f}")
    return test_loss

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

def test_lags(dataloader, model, loss_fn, device):
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
        for batch, (D,I,P) in enumerate(dataloader):
            pred, I_pred = model(P.to(device))
            pred, I_pred = pred[:,None,:], I_pred[:,None,:]
            test_loss += loss_fn(pred, I_pred, D.to(device), I.to(device)).detach().item()
    test_loss /= batches
    
    print(f"Average testing loss: {test_loss:>8f}")
    return test_loss

def active_training_loop(model,dataloader,optimizer,loss_fn,device,
                        test_dataloader,te_loss_arr,tr_loss_arr,
                        last_sig_te,last_sig_tr, active_loop_num,
                        loop_epochs, best_model, train, test,
                        mode = "flux", stopping = 15, scheduler = None):
    epoch = 0
    #set improvements counters to 0
    imp_te = 0
    imp_tr = 0
    
    print(f"Training {mode} model")
    while (imp_te < stopping or imp_tr < stopping):
        print(f"Epoch {epoch+1} \n -----------------------")
        
        model, optimizer, train_loss = train(dataloader, model,
                                             optimizer, loss_fn,
                                             device)
        loss = test(test_dataloader, model, loss_fn, device)
        if scheduler != None:
            scheduler.step(loss)
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
            torch.save(model.state_dict(), f"models/active_best_{mode}.pth")
        
        epoch += 1
    
    if active_loop_num != 0:
        loop_epochs.append(loop_epochs[active_loop_num-1]+epoch)
    else:
        loop_epochs.append(epoch)
        
    mergeSaveData(pd.read_csv(f"data/locations/active_test_locs_{mode}.csv"), 
                  pd.read_csv(f"data/locations/active_locs_{mode}.csv"), 
                  "data/locations/",f"active_locs_{mode}.csv")
    
    temp_te = np.asarray(te_loss_arr)
    temp_tr = np.asarray(tr_loss_arr)
    temp_epochs = np.asarray(loop_epochs)
    try:
        best_model.load_state_dict(torch.load(f"models/active_best_{mode}.pth"))
    except:
        best_model.load_state_dict(model.state_dict())
        
    saveLoop(best_model, f"data/locations/active_locs_{mode}.csv", 
             optimizer,
             temp_te, temp_tr, 
             active_loop_num, temp_epochs,
             typ=mode)
    
    return (model, best_model, optimizer, loop_epochs, 
            te_loss_arr, tr_loss_arr, last_sig_tr, last_sig_te)

def grid_training_loop(model, optimizer, train, test, train_dataloader, 
                       test_dataloader, loss_fn, device, name, mode, 
                       epochs = 400, scheduler = None):
    tr_loss_arr = []
    med_tr_loss_arr = []
    te_loss_arr = []
    std_tr_loss_arr = []
    
    epoch = 0
    imp_flag = 0
    print("Beginning training")
    
    while epoch < epochs and imp_flag < 150:
        imp_flag += 1
        #time_st = time.time()
        print(f"Epoch {epoch+1} \n -----------------------")
        model,optimizer,train_loss,med_loss,std_loss = train(train_dataloader, 
                                                             model,
                                             optimizer, loss_fn, device,
                                             scheduler,epoch)
        loss = test(test_dataloader, model, loss_fn, device)
        te_loss_arr.append(loss)
        tr_loss_arr.append(train_loss)
        med_tr_loss_arr.append(med_loss)
        std_tr_loss_arr.append(std_loss)
        if loss == np.min(te_loss_arr):
            print(f"New best testing loss: {loss}")
            torch.save(model.state_dict(), f"models/{name}_{mode}.pth")
            imp_flag = 0
        epoch += 1
    
    print("Completed training")
    print("Final best training loss:", np.min(tr_loss_arr))
    print("Final best testing loss:", np.min(te_loss_arr))
    torch.save(model.state_dict(), f"models/{name}_{mode}_final.pth")
    print(f"Saved PyTorch Model State to {name}_{mode}_final.pth")
    
    tr_loss_arr = np.asarray(tr_loss_arr)
    te_loss_arr = np.asarray(te_loss_arr)
    med_tr_loss_arr = np.asarray(med_tr_loss_arr)
    std_tr_loss_arr = np.asarray(std_tr_loss_arr)
    
    np.savetxt(f"loss/{name}_{mode}_te_loss.txt",te_loss_arr)
    np.savetxt(f"loss/{name}_{mode}_tr_loss.txt",tr_loss_arr)
    np.savetxt(f"loss/{name}_{mode}_med_tr_loss.txt",med_tr_loss_arr)
    np.savetxt(f"loss/{name}_{mode}_std_tr_loss.txt",std_tr_loss_arr)
    
    return

def bottleneck_training_loop(model, optimizer, train_dataloader, 
                       test_dataloader, loss_fn, device, name, mode, 
                       epochs = 400):
    tr_loss_arr = []
    te_loss_arr = []
    
    pars = 10
    
    print("Beginning training")
    for par in range(pars):
        epoch = 0
        mask = torch.ones(pars)
        mask[par+1:] = 0
        print(f"Training on {par+1} parameters")
        print(f"Currently, the mask is {mask}")
        while (epoch < epochs):
            print(f"Epoch {epoch+1} \n -----------------------")
            model, optimizer, train_loss = train_bottle(train_dataloader, model,
                                                 optimizer, loss_fn, device,
                                                 mask)
            loss = test_bottle(test_dataloader, model, loss_fn, device,mask)
            te_loss_arr.append(loss)
            tr_loss_arr.append(train_loss)
            if loss == np.min(te_loss_arr):
                print(f"New best testing loss: {loss}")
                torch.save(model.state_dict(), f"models/{name}_{mode}.pth")
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

def QBDC(flux_name, flux_test_name, lags_name, lags_test_name, active_loop_num, 
         theta_lhc, lhc_idx, egrid, lags_egrid, flux_model, lags_model, device,
         labels, parallel):
    #retrieves output features shape
    module_list = [module for module in flux_model.modules()]
    flux_out = module_list[-1].out_features
    module_list = [module for module in lags_model.modules()]
    lags_out = module_list[-1].out_features
    
    data_size = len(pd.read_csv(f"data/locations/{flux_name}"))
    multiplier = ceil(data_size/100000)
    n_samples = 5000*multiplier
    n_samples_large = 50000*multiplier # number of parameter sets to draw 
    divider = 100*multiplier
    n_samples_small = int(n_samples_large/divider)
    print(f"I am in active learning loop {active_loop_num}")
    # randomly generate points in parameter space
    print("Generating random samples of theta")
    theta_query_large = theta_lhc[lhc_idx : lhc_idx+n_samples_large]
    
    print("computing neural network predictions with dropout for each theta")
    # compute 100 neural network predictions with dropout
    sample_dropout = 100
    pred_query_flux = np.zeros((sample_dropout,n_samples_small,flux_out))
    pred_query_lags = np.zeros((sample_dropout,n_samples_small,lags_out))
    flux_model.train()
    lags_model.train()
    query_samples = []
    
    for j in tqdm(range(divider),desc="Sample dropout loops"):
        theta_query_small = theta_query_large[j*n_samples_small:(j+1)*n_samples_small]
        for i in range(sample_dropout):
            pred_flux = flux_model(torch.FloatTensor(theta_query_small).to(device))
            pred_lags = lags_model(torch.FloatTensor(theta_query_small).to(device))
            pred_query_flux[i] = pred_flux.detach().cpu().numpy()
            pred_query_lags[i] = pred_lags.detach().cpu().numpy()
        # find uncertainty (as measured by relative variance)
        dvar_flux = np.var(pred_query_flux,axis=0)
        mean_var_flux = np.mean(dvar_flux, axis=1)
        # find uncertainty (as measured by relative variance)
        dvar_lags = np.var(pred_query_lags,axis=0)
        mean_var_lags = np.mean(dvar_lags, axis=1)
        #sum the two
        mean_var_query = mean_var_flux+mean_var_lags
        # add to uncertainties per theta to list
        query_samples.append(mean_var_query.tolist())
    
    #Performing manual memory cleanup
    print("Successfully finished generating thetas")
    
    print("Finding top uncertain thetas")
    # sort these thetas from smallest uncertainty to largest and save values
    query_samples = np.asarray(query_samples).flatten()
    np.savetxt(f"dists/loop_{active_loop_num}_variances.txt",query_samples)
    query_idx = np.argsort(query_samples)[::-1]
    
    print("Generating data for these samples")
    # get out the top `nsamples` values of theta_query
    theta_query = theta_query_large[query_idx[:n_samples]]
    
    active_learning_generation(theta_query, egrid, lags_egrid, parallel, 
                                flux_name, flux_test_name, lags_name,
                                lags_test_name)
    
    # add rejected parameter sets back to original array for potential 
    # future use:
    theta_lhc = np.vstack([theta_lhc, theta_query_large[query_idx[n_samples:]]])
    
    # increment the index for reading parameters from theta_lhc
    lhc_idx += (n_samples_large)
    
    return theta_lhc, lhc_idx
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
from math import ceil
from tqdm import tqdm
from plotting import distributions, variances
from generator import active_learning_generation, pars_conversion

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
        self.min = torch.tensor(self.scaler.data_min_)
        self.max = torch.tensor(self.scaler.data_max_)
        self.scale = self.max - self.min
    
    def scaling(self,a):
        result = (a * self.scale.to(self.device)) + self.min.to(self.device)
        result = 10**result
        return result
    
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
    
class magDecFluxLoss(nn.Module):
    """
    Class of loss functon that produces weighted mean square error loss for
    NN outputs with outputs x and y where the true data is of the form 
    x * 10^y.
    
    -------------
    Parameters:
        dec_scaler:
            loads a MinMaxScaler from scikitlearn that allows retrieval of
            original decimal values
        mag_scaler:
            loads a MinMaxScaler that allows scaling of magnitudes to original
            values
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
        
        scaling(dec, mag):
            dec is the decimal value of the output to be scaled while mag
            is the corresponding order of magnitude
        
        forward(output, target):
            forward pass that returns a loss value based on the weighted mean 
            squared error and masking where outputs are below the threshold
            value.
    """
    def __init__(self,mag_scaler,device):
        super().__init__()
        self.mag_scaler = load(f'scalers/{mag_scaler}')
        self.device = device
        self.threshold = 1e-39
        self.criterion = nn.MSELoss()
        self.set_scale()
        
    def set_scale(self):
        self.dec_min = torch.tensor(1)
        self.dec_scale = torch.tensor(9)
        self.mag_min = torch.tensor(self.mag_scaler.data_min_)
        self.mag_scale = torch.tensor(self.mag_scaler.data_range_)
        self.mag_scale_mask = torch.where(self.mag_scale == 0,0,1)
        
    def scaling(self,dec,mag):
        dec_sca = (dec * self.dec_scale.to(self.device)) + self.dec_min.to(self.device)
        mag_sca = torch.floor((mag * self.mag_scale.to(self.device)) + self.mag_min.to(self.device))
        result = dec_sca * 10**mag_sca
        return result
    
    def normalize(self,dec,mag):
        dec = ((dec - self.dec_min.to(self.device))/self.dec_scale.to(self.device))
        mag = ((mag - self.mag_min.to(self.device))/self.mag_scale.to(self.device))
        mag[:,:,self.mag_scale_mask.to(self.device)==0] = 0
        return dec, mag
        
    def forward(self, output_dec, output_mag, target, test = False):
        target_mag = torch.floor(torch.log10(target))
        target_dec = target/10**target_mag
        target_dec, target_mag = self.normalize(target_dec, target_mag)
        #scale to real space
        scaled_out = self.scaling(output_dec,output_mag)
        mask = torch.where((target<=self.threshold)&(scaled_out<=self.threshold),0,1)
        #set both values in tensors to 1 where mask is equal to 0
        target_mag = torch.mul(target_mag,mask)
        target_dec = torch.mul(target_dec,mask)
        output_mag = torch.mul(output_mag,mask)
        output_dec = torch.mul(output_dec,mask)
        #calculate loss
        mag_loss = self.criterion(output_mag,target_mag)
        dec_loss = self.criterion(output_dec,target_dec)
        loss = mag_loss + dec_loss
        return loss 

class magDecLagsLoss(nn.Module):
    """
    Class of loss functon that produces weighted mean square error loss for
    NN outputs with outputs x, y and i where the true data is of the form 
    i*x * 10^y.
    
    -------------
    Parameters:
        dec_scaler:
            loads a MinMaxScaler from scikitlearn that allows retrieval of
            original decimal values
        mag_scaler:
            loads a MinMaxScaler that allows scaling of magnitudes to original
            values
        device:
            pytorch device to load tensors into for manipulation
    
    -------------
    Methods:
        __init__():
            initalizes key parameters
        
        set_scale():
            used in intializing to calculate rescaling method
        
        scaling(dec, mag):
            dec is the decimal value of the output to be scaled while mag
            is the corresponding order of magnitude
        
        forward(output_dec, output_mag, target_dec, target_mag, 
                    output_ind, target_ind):
            forward pass that returns a loss value. This loss is based on the 
            weighted mean squared error of the true data vs NN output 
            and masking where outputs are below the threshold as well as adding
            the binary cross entropy loss for whether the NN correctly 
            identified the lag as negative or positive with 1 representing
            positive lags and 0 representing negative lags.
    """
    def __init__(self,mag_scaler,device):
        super().__init__()
        self.mag_scaler = load(f'scalers/{mag_scaler}')
        self.device = device
        self.threshold = 1e-7
        self.binary = nn.BCELoss()
        self.criterion = nn.MSELoss()
        self.set_scale()
        
    def set_scale(self):
        self.dec_min = torch.tensor(1)
        self.dec_scale = torch.tensor(9)
        self.mag_min = torch.tensor(self.mag_scaler.data_min_)
        self.mag_scale = torch.tensor(self.mag_scaler.data_range_)
        self.mag_scale_mask = torch.where(self.mag_scale == 0,0,1)
        
    def scaling(self,dec,mag):
        dec_sca = (dec * self.dec_scale.to(self.device)) + self.dec_min.to(self.device)
        mag_sca = torch.floor((mag * self.mag_scale.to(self.device)) + self.mag_min.to(self.device))
        result = dec_sca * 10**mag_sca
        return result
    
    def normalize(self,dec,mag):
        dec = ((dec - self.dec_min.to(self.device))/self.dec_scale.to(self.device))
        mag = ((mag - self.mag_min.to(self.device))/self.mag_scale.to(self.device))
        mag[:,:,self.mag_scale_mask.to(self.device)==0] = 0
        return dec, mag
        
    def forward(self, output_dec, output_mag, output_ind, target, target_ind, 
                test = False):
        target_mag = torch.floor(torch.log10(target))
        target_dec = target/10**target_mag
        target_dec, target_mag = self.normalize(target_dec, target_mag)
        #scale to real space
        scaled_out = self.scaling(output_dec,output_mag)
        mask = torch.where((target <= self.threshold)&(scaled_out<=self.threshold),0,1)
        #set both values in tensors to 1 where mask is equal to 0
        target_mag = torch.mul(target_mag,mask)
        target_dec = torch.mul(target_dec,mask)
        output_mag = torch.mul(output_mag,mask)
        output_dec = torch.mul(output_dec,mask)
        #calculate loss
        mag_loss = self.criterion(output_mag,target_mag)
        dec_loss = self.criterion(output_dec,target_dec)
        signed_loss = self.binary(output_ind,target_ind)
        loss = signed_loss + mag_loss + dec_loss
        return loss
    
def train_flux(dataloader, model, optimizer, loss_fn,device, dec_mag = False):
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
        if torch.any(torch.isnan(D)) == True:
            print("D contains NaNs")
            print(D)
            quit()
        elif torch.any(torch.isinf(D)) == True:
            print("D contains infinities")
            print(D)
            quit()
        if torch.any(torch.isnan(P)) == True:
            print("P contains NaNs")
            print(P)
            quit()
        elif torch.any(torch.isinf(P)) == True:
            print("D contains infinities")
            print(P)
            quit()
        optimizer.zero_grad()
        for name, param in model.named_parameters():
            if torch.any(torch.isnan(param.data)) == True:
                print(f"Param is {param.data}")
                print("Exiting program")
                quit()
            elif torch.any(torch.isinf(param.data)) == True:
                print(f"Param is {param.data}")
                print("Exiting program")
                quit()
        if dec_mag == False:
            pred = model(P.to(device))[:,None,:]
            loss = loss_fn(pred,D.to(device))
        else:
            mag_pred, dec_pred  = model(P.to(device))
            mag_pred, dec_pred = mag_pred[:,None,:], dec_pred[:,None,:]
            loss = loss_fn(dec_pred, mag_pred, D.to(device))
        loss.backward()
        #prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        loss_b = loss.detach().item()
        if batch % 5 == 0:
            current = (batch*P.shape[0] + 1)
            print(f"loss: {loss_b:>7f}  [{current:>5d}/{size:>5d}]")
        loss_arr += loss_b
    
    avg_loss = loss_arr/len(dataloader)
    print(f"Average training loss: {avg_loss:>8f}")
    return model, optimizer , avg_loss

def train_lags(dataloader, model, optimizer, loss_fn, device, dec_mag=False):
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
    for batch, (D, I, P) in enumerate(dataloader):
        for name, param in model.named_parameters():
            if torch.any(torch.isnan(param.data)) == True:
                print(f"Param is {param.data}")
                print("Exiting program")
                quit()
            elif torch.any(torch.isinf(param.data)) == True:
                print(f"Param is {param.data}")
                print("Exiting program")
                quit()
        optimizer.zero_grad()
        if dec_mag == False:
            pred, I_pred = model(P.to(device))
            pred, I_pred = pred[:,None,:], I_pred[:,None,:]
            loss = loss_fn(pred, I_pred, D.to(device), I.to(device))
        else:
            mag_pred, dec_pred, I_pred = model(P.to(device))
            mag_pred, dec_pred, I_pred = mag_pred[:,None,:], dec_pred[:,None,:], I_pred[:,None,:]
            loss = loss_fn(dec_pred, mag_pred, I_pred, D.to(device), I.to(device))
        loss.backward()
        #prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_b = loss.detach().item()
        if batch % 5 == 0:
            current = (batch*P.shape[0] + 1)
            print(f"loss: {loss_b:>7f}  [{current:>5d}/{size:>5d}]")
        loss_arr += loss_b
        
    avg_loss = loss_arr/len(dataloader)
    print(f"Average training loss: {avg_loss:>8f}")
    return model, optimizer , avg_loss

def test_flux(dataloader, model, loss_fn, device, dec_mag=False):
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
            if dec_mag == False:
                pred = model(P.to(device))[:,None,:]
                test_loss += loss_fn(pred,D.to(device)).detach().item()
            else:
                mag_pred, dec_pred = model(P.to(device))
                mag_pred, dec_pred = mag_pred[:,None,:], dec_pred[:,None,:]
                test_loss += loss_fn(dec_pred, mag_pred, D.to(device), test=True).detach().item()
    test_loss /= batches
    
    print(f"Average testing loss: {test_loss:>8f}")
    return test_loss

def test_lags(dataloader, model, loss_fn, device, dec_mag=False):
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
            if dec_mag == False:
                pred, I_pred = model(P.to(device))
                pred, I_pred = pred[:,None,:], I_pred[:,None,:]
                test_loss += loss_fn(pred, I_pred, D.to(device), I.to(device)).detach().item()
            else:
                mag_pred, dec_pred, I_pred = model(P.to(device))
                mag_pred, dec_pred, I_pred = mag_pred[:,None,:], dec_pred[:,None,:], I_pred[:,None,:]
                test_loss += loss_fn(dec_pred, mag_pred, I_pred, D.to(device), 
                                     I.to(device), test=True).detach().item()
    test_loss /= batches
    
    print(f"Average testing loss: {test_loss:>8f}")
    return test_loss

def active_training_loop(model,dataloader,optimizer,loss_fn,device,
                        test_dataloader,te_loss_arr,tr_loss_arr,
                        last_sig_te,last_sig_tr, active_loop_num,
                        loop_epochs, best_model, train, test,
                        mode = "flux", stopping = 15, scheduler = None,
                        dec_mag=False):
    epoch = 0
    #set improvements counters to 0
    imp_te = 0
    imp_tr = 0
    
    print(f"Training {mode} model")
    while (imp_te < stopping or imp_tr < stopping):
        print(f"Epoch {epoch+1} \n -----------------------")
        
        model, optimizer, train_loss = train(dataloader, model,
                                             optimizer, loss_fn,
                                             device, dec_mag)
        loss = test(test_dataloader, model, loss_fn, device, dec_mag)
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
                       test_dataloader, loss_fn, device, size, mode, 
                       epochs = 400, dec_mag = False):
    
    last_sig_best_tr = 1e7 #last significant best training loss (set large initially)
    last_sig_best_te = 1e7 #last significant best testing loss (set large initially)
    tr_loss_arr = []
    te_loss_arr = []
    
    epoch = 0
    imp_te = 0
    imp_tr = 0
    
    print("Beginning training")
    while epoch < epochs:
        print(f"Epoch {epoch+1} \n -----------------------")
        model, optimizer, train_loss = train(train_dataloader, model,
                                             optimizer, loss_fn, device,
                                             dec_mag=dec_mag)
        loss = test(test_dataloader, model, loss_fn, device, dec_mag=dec_mag)
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
            torch.save(model.state_dict(), f"models/grid_{size}_{mode}.pth")
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
            torch.save(model.state_dict(), f"models/grid_{size}_{mode}.pth")
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
    
    return

def QBDC(flux_name, flux_test_name, lags_name, lags_test_name, active_loop_num, 
         theta_lhc, lhc_idx, egrid, lags_egrid, flux_model, lags_model, 
         dec_mag, device, labels, parallel):
    data_size = len(pd.read_csv(f"data/locations/{flux_name}"))
    multiplier = ceil(data_size/100000)
    n_samples = 500*multiplier
    n_samples_large = 1000*multiplier # number of parameter sets to draw 
    divider = 100*multiplier
    n_samples_small = int(n_samples_large/divider)
    print(f"I am in active learning loop {active_loop_num}")
    # randomly generate points in parameter space
    print("Generating random samples of theta")
    theta_query_large = theta_lhc[lhc_idx : lhc_idx+n_samples_large]
    
    print("computing neural network predictions with dropout for each theta")
    # compute 100 neural network predictions with dropout
    sample_dropout = 100
    pred_query_flux = np.zeros((sample_dropout,n_samples_small,len(egrid)))
    pred_query_lags = np.zeros((sample_dropout,n_samples_small,len(lags_egrid)-1))
    pred_query_inds = np.zeros((sample_dropout,n_samples_small,len(lags_egrid)-1))
    flux_model.train()
    lags_model.train()
    query_samples = []
    
    for j in tqdm(range(divider),desc="Sample dropout loops"):
        theta_query_small = theta_query_large[j*n_samples_small:(j+1)*n_samples_small]
        for i in range(sample_dropout):
            if dec_mag == True:
                mag_flux, dec_flux = flux_model(torch.DoubleTensor(theta_query_small).to(device))
                mag_lags, dec_lags, ind = lags_model(torch.DoubleTensor(theta_query_small).to(device))
                pred_query_flux[i] = mag_flux.detach().cpu().numpy() + dec_flux.detach().cpu().numpy()
                pred_query_lags[i] = mag_lags.detach().cpu().numpy() + dec_lags.detach().cpu().numpy()
                pred_query_inds[i] = ind.detach().cpu().numpy()
            else:
                pred_flux = flux_model(torch.DoubleTensor(theta_query_small).to(device))
                pred_lags, ind = lags_model(torch.DoubleTensor(theta_query_small).to(device))
                pred_query_flux[i] = pred_flux.detach().cpu().numpy()
                pred_query_lags[i] = pred_lags.detach().cpu().numpy()
                pred_query_inds[i] = ind.detach().cpu().numpy()
        # find uncertainty (as measured by relative variance)
        dvar_flux = np.var(pred_query_flux,axis=0)
        mean_var_flux = np.mean(dvar_flux, axis=1)
        # find uncertainty (as measured by relative variance)
        dvar_lags = np.var(pred_query_lags,axis=0)
        mean_var_lags = np.mean(dvar_lags, axis=1)
        # find uncertainty (as measured by relative variance)
        dvar_inds = np.var(pred_query_inds,axis=0)
        mean_var_inds = np.mean(dvar_inds, axis=1)
        #sum the two
        mean_var_query = mean_var_flux+0.5*(mean_var_lags+mean_var_inds)
        # add to uncertainties per theta to list
        query_samples.append(mean_var_query.tolist())
    
    #Performing manual memory cleanup
    print("Successfully finished generating thetas")
    
    print("Finding top uncertain thetas")
    # sort these thetas from smallest uncertainty to largest and save values
    query_samples = np.asarray(query_samples).flatten()
    np.savetxt(f"dists/loop_{active_loop_num}_variances.txt",query_samples)
    query_idx = np.argsort(query_samples)[::-1]
    
    variances(query_samples, active_loop_num)
    
    print("Top sample mean variance",query_samples[query_idx[0]])
    print("Bottom sample mean variance",query_samples[query_idx[-1]])
    print("Range of mean variance",np.ptp(query_samples))
    
    print("Generating data for these samples")
    # get out the top `nsamples` values of theta_query
    theta_query = theta_query_large[query_idx[:n_samples]]
    
    distributions(theta_query, labels, f"dists/loop_{active_loop_num}_")
    
    active_learning_generation(theta_query, egrid, lags_egrid, parallel, 
                                   flux_name, flux_test_name, lags_name,
                                   lags_test_name, pars_conversion)
    
    # add rejected parameter sets back to original array for potential 
    # future use:
    theta_lhc = np.vstack([theta_lhc, theta_query_large[query_idx[n_samples:]]])
    
    # increment the index for reading parameters from theta_lhc
    lhc_idx += (n_samples_large)
    
    return theta_lhc, lhc_idx
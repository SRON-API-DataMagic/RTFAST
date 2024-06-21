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
        assert torch.any(torch.isnan(D)) == False,"NaNs present in data"
        assert torch.any(torch.isnan(P)) == False,"NaNs present in pars"
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
    
    temp_te = np.asarray(te_loss_arr)
    temp_tr = np.asarray(tr_loss_arr)
    temp_epochs = np.asarray(loop_epochs)
    try:
        best_model.load_state_dict(torch.load(f"models/active_best_{mode}.pth"))
    except:
        best_model.load_state_dict(model.state_dict())
        
    saveLoop(best_model, "data/locations/locs_active_tra.csv", optimizer,
             temp_te, temp_tr, active_loop_num, temp_epochs)
    
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

def QBDC(flux_name, flux_test_name, active_loop_num,
         pool_csv, val_csv, flux_model, device, labels):
    #retrieves output features shape
    module_list = [module for module in flux_model.modules()]
    flux_out = module_list[-1].out_features
    
    data_size = len(pd.read_csv("data/locations/locs_active_tra.csv"))
    multiplier = ceil(data_size/100000)
    n_samples = 5000*multiplier
    n_samples_large = 500000*multiplier # number of parameter sets to draw 
    if n_samples_large > 5*10**6:
        n_samples_large = 5*10**6
    divider = 100*multiplier
    n_samples_small = int(n_samples_large/divider)
    print(f"I am in active learning loop {active_loop_num}")
    # randomly generate points in parameter space
    print("Selecting random parameter sets")
    
    pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
    negatives = [3]
    logged = [0,2,3,4,7,8,10,11,12,13,23]
    
    theta_lhc = pool_csv.iloc[:,pars_list].to_numpy()
    
    for i, parameter in enumerate(pars_list):
        if parameter in logged:
            theta_lhc[:,i] = 10*theta_lhc[:,i]
        else:
            theta_lhc[:,i] = theta_lhc[:,i]
        if parameter in negatives:
            theta_lhc[:,i] = -theta_lhc[:,i]

    theta_query_large = theta_lhc[:n_samples_large]
    
    print("Computing neural network predictions with dropout for each theta")
    # compute 100 neural network predictions with dropout
    sample_dropout = 100
    pred_query_flux = np.zeros((sample_dropout,n_samples_small,flux_out))
    flux_model.train()
    query_samples = []
    
    for j in tqdm(range(divider),desc="Sample dropout loops"):
        theta_query_small = theta_query_large[j*n_samples_small:(j+1)*n_samples_small]
        for i in range(sample_dropout):
            pred_flux = flux_model(torch.FloatTensor(theta_query_small).to(device))
            pred_query_flux[i] = pred_flux.detach().cpu().numpy()
        # find uncertainty (as measured by relative variance)
        dvar_flux = np.var(pred_query_flux,axis=0)
        mean_var_flux = np.mean(dvar_flux, axis=1)
        # add to uncertainties per theta to list
        query_samples.append(mean_var_flux.tolist())
    
    #Performing manual memory cleanup
    print("Successfully finished generating thetas")
    
    print("Finding top uncertain thetas")
    # sort these thetas from smallest uncertainty to largest and save values
    query_samples = np.asarray(query_samples).flatten()
    np.savetxt(f"dists/loop_{active_loop_num}_variances.txt",query_samples)
    query_idx = np.argsort(query_samples)[::-1]
    
    print("Generating data for these samples")
    # get out the top `nsamples` values of theta_query and add old val data
    new_data = pool_csv.iloc[query_idx[int(0.1*n_samples):n_samples]]
    total_new_data = pd.concat([new_data,val_csv],ignore_index=True)
    val_csv = pool_csv.iloc[query_idx[:int(0.1*n_samples)]]
    #add newly selected data to theta
    mergeSaveData(total_new_data, 
                  pd.read_csv("data/locations/locs_active_tra.csv"), 
                  "data/locations/", "locs_active_tra.csv")
    
    # add rejected parameter sets back to original array for potential 
    # future use:
    pool_csv = pd.concat([pool_csv,pool_csv.iloc[query_idx[n_samples:]]],
                         ignore_index=True)
    #remove parameter sets checked
    pool_csv = pool_csv.iloc[n_samples_large:]
    #reset index
    pool_csv.reset_index(inplace=True,drop=True)
    val_csv.to_csv("locs_active_val.csv",ignore_index=True)
    
    return pool_csv
"""
This program holds the base training and testing processes for neural network
training.
"""

import torch
from torch import nn
from joblib import load

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

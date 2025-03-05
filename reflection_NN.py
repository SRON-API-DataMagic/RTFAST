"""
Gross test program
"""

from network import DynamicNetwork
from training import PCALoss, train_flux, test_flux, training_loop
from joblib import load, dump
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from dataStructures import DataStructure

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pars_list = [0,1,2,3,4,6,7,8,9,10]
    negatives = [3]
    logged = [0,2,3,4,10]
    
    pars = np.loadtxt("data/pars.txt")
    pars = pars[:,pars_list]
    for i, parameter in enumerate(pars_list):
        if parameter in negatives:
            pars[:,i] = -pars[:,i]
        if parameter in logged:
            pars[:,i] = np.log10(pars[:,i])
    
    spectra = np.loadtxt("data/spectra.txt")
    mask = np.any(spectra <= 0,axis=1)
    spectra[spectra<=0] = 1e-11
    spectra = spectra[~mask]
    pars = pars[~mask]
    mask = np.any(np.isnan(spectra),axis=1)
    spectra = spectra[~mask]
    pars = pars[~mask]
    
    pca = load("scalers/pca.bin")
    scaler = load("scalers/scaler.bin")
    pca_scaler = load("scalers/pca_scaler.bin")
    
    data = pca.transform(scaler.transform(np.log10(spectra)))
    data = pca_scaler.transform(data)
    
    split = int(0.9*len(data))
    
    batch_size = 1024
    assert len(pars) == len(data)
    
    train_data = DataStructure(pars[:split], data[:split])
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True)
    validation_data = DataStructure(pars[split:], data[split:])
    validation_loader = DataLoader(validation_data, batch_size=batch_size,
                                   shuffle=True)
    
    loss_fn = PCALoss(pca.explained_variance_ratio_, device)
    
    train = train_flux
    test =  test_flux
    for layers in [4,6,8]:
        model = DynamicNetwork(len(pars_list),200,layers,256)
        model.to(device)
        model.double()
        optimizer = Adam(model.parameters(), lr=1e-3)
        #optimizer = SGD(model.parameters(), lr=1e-4,momentum=0.9)
        training_loop(model, optimizer, train, test, train_loader,
                      validation_loader, loss_fn, device, f"reflection_{layers}",
                      epochs = 1000)
    
if __name__ == "__main__":
    main()
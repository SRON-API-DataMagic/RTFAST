"""
This is a quick training exercise to test if using PCA on spectra and using very
lightweight NNs is a viable alternative to what we've been doing up until now.
"""

from generator import intialize_dataset, rtdist_flux, nn_pars_to_rtdist
import generator
from training import grid_training_loop, train_flux, test_flux, PCALoss
import numpy as np
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sherpa.astro.ui import unpack_rmf
from processing import saveData, spectraChecker

from joblib import Parallel, delayed, dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from dataStructures import PCADataset
from network import PCAFluxNetwork

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

wrk_dir = os.getcwd()

rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
rmf = unpack_rmf(rmf_name)
egrid = rmf.e_min #energy grid used to evaluate the xspec model

pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
negatives = [3]
logged = [0,2,3,4,7,8,10,11,12,13,23]

pars_list = [1,2,3,4,13]
negatives = [3]
logged = [2,3,4,13]

range_AGN = np.asarray(generator.lhc_trimmed_gen())
num_pars = range_AGN.shape[0]
"""
theta_lhc = generator.lhc_generation(int(1e5), range_AGN, limited=True)

flux_name = "PCA_locs_flux.csv"
flux_test_name = "PCA_test_locs_flux.csv"
flux_scaler_name = "PCA_scaler_flux.bin"

#generate physical models of test set
theta_flux = nn_pars_to_rtdist(theta_lhc, 0, pars_list, negatives, logged)
theta_lags = nn_pars_to_rtdist(theta_lhc, 0, pars_list, negatives, logged)
print("Parallelized model generation")

print("Generating flux models")
flux =  Parallel(n_jobs=20,verbose=5)(delayed(rtdist_flux)(pars, egrid)
                                for pars in theta_flux)
flux = np.asarray(flux)
print("Checking for spectra below threshold")
flux, theta_flux, theta_lags = spectraChecker(flux,theta_flux,theta_lags,
                                              1e-11)

idxs = np.arange(0,flux.shape[0])
np.random.shuffle(idxs)
tra_idx = idxs[:int(0.9*len(idxs))]
tes_idx = idxs[int(0.9*len(idxs)):]

#Splitting data and parameters into training and testing datasets
train_flux_data = flux[tra_idx]
train_flux_pars = theta_flux[tra_idx]

test_flux_data = flux[tes_idx]
test_flux_pars = theta_flux[tes_idx]

print("Saving flux data")
saveData(train_flux_data, train_flux_pars, 
         "data/locations/","PCA_locs_flux.csv")
saveData(test_flux_data, test_flux_pars, 
         "data/locations/","PCA_locs_flux_test.csv")
"""
train_dataset = PCADataset("data/locations/PCA_locs_flux.csv",
                           pars_list,negatives,logged,scale_bool = False,
                           comps=5)
train_dataloader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                              shuffle=True)

test_dataset = PCADataset("data/locations/PCA_locs_flux_test.csv",
                           pars_list,negatives,logged,scale_bool = False)
test_dataloader = DataLoader(test_dataset, batch_size=1024, num_workers = 4, 
                              shuffle=True)

model = PCAFluxNetwork(num_pars, train_dataset.data.shape[1])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

model.float()

optimizer = Adam(model.parameters(),lr = 0.001)

loss_fn = PCALoss()

train = train_flux
test =  test_flux

grid_training_loop(model, optimizer, train, test, train_dataloader, 
                   test_dataloader, loss_fn, device, "PCA", "flux", 
                   epochs = 3000)
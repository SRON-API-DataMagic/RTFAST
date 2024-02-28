"""
This is a quick training exercise to test if using PCA on spectra and using very
lightweight NNs is a viable alternative to what we've been doing up until now.
"""

from generator import intialize_dataset, rtdist_flux, nn_pars_to_rtdist
import generator
from training import grid_training_loop, train_flux, test_flux, PCALoss
from training import bottleneck_training_loop
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
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

wrk_dir = os.getcwd()

rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
rmf = unpack_rmf(rmf_name)
egrid = rmf.e_min #energy grid used to evaluate the xspec model
egrid = egrid[egrid>0.1]

#set envionmental variables required in xspec with simrtdist
environ_vars = {"REV_VERB":"0","MU_ZONES":"1","ION_ZONES":"1","A_DENSITY":"1",
                "EMIN_REF":"0.5","EMAX_REF":"10","EMIN_REF2":"0.5",
                "EMAX_REF2":"10", "SEED_SIM":"-2851043",
                "RMF_SET":wrk_dir+"/ResponseFiles/PN.rmf",
                "ARF_SET":wrk_dir+"/ResponseFiles/PN.arf",
                "BKG_SET":wrk_dir+"/ResponseFiles/PNbackground_spectrum.fits",
                "BACKSCL":"1.0"}

for key in environ_vars:
    os.environ[key] = environ_vars[key]
"""
pars_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,23]
negatives = [3]
logged = [0,2,3,4,7,8,10,11,12,13,23]

"""
pars_list = [1,2,3,6,7,8,9,11,13,23]
negatives = [3]
logged = [2,3,7,8,11,13,23]
"""
pars_list = [1,2,3,4,13]
negatives = [3]
logged = [2,3,4,13]
"""
"""
pars_list = [3]
negatives = [3]
logged = [3]
"""
range_AGN = np.asarray(generator.lhc_10())
num_pars = len(pars_list)
print(num_pars)

theta_lhc = generator.lhc_generation(int(1e5), range_AGN, limited=False, 
                                     lhc_filter=generator.lhc_filter_10)

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


val_dataset = PCADataset("data/locations/PCA_locs_flux_test.csv",
                           pars_list,negatives,logged,scale_bool = True,
                           force=True,comps=40,
                           threshold=1e-11)
val_dataloader = DataLoader(val_dataset, batch_size=1024, num_workers = 4, 
                              shuffle=True)

train_dataset = PCADataset("data/locations/PCA_locs_flux.csv",
                           pars_list,negatives,logged,scale_bool = False,
                           threshold=1e-11)
train_dataloader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                              shuffle=True)
print(val_dataset.data.shape)
model = PCAFluxNetwork(num_pars, val_dataset.data.shape[1])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

model.float()

optimizer = Adam(model.parameters(),lr = 1e-3)
#optimizer = SGD(model.parameters(),lr=1e-2,momentum=0.9)

loss_fn = PCALoss(val_dataset.pca.explained_variance_ratio_, device)
#loss_fn = nn.MSELoss()
"""
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, 
                                              max_lr=1e-2)
"""
train = train_flux
test =  test_flux

grid_training_loop(model, optimizer, train, test, train_dataloader, 
                   val_dataloader, loss_fn, device, "PCA", "flux", 
                   epochs = 1000)

"""
bottleneck_training_loop(model, optimizer,train_dataloader, 
                   val_dataloader, loss_fn, device, "PCA", "flux", 
                   epochs = 500)
"""
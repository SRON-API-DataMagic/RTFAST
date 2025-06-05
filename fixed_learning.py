"""
This is a quick training exercise to test if using PCA on spectra and using very
lightweight NNs is a viable alternative to what we've been doing up until now.
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from joblib import dump, load, Parallel, delayed

from dataStructures import PCAtrimmedDataset
from network import DynamicNetwork, DynamicResNetwork
from training import training_loop, train, validate, PCALoss
from tqdm import tqdm
import numpy as np
import glob

def file_load(file):
    return np.loadtxt(file)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pars_list = [0,1,2,3,4,6,7,8,9,10]
    negatives = [3]
    logged = [0,2,3,4,10]
    
    data_locs = np.sort(glob.glob("data/pca_comps/pca_comps_*.txt"))
    pars_locs = np.sort(glob.glob("data/pars/pars_*.txt"))
    with Parallel(n_jobs=-1,verbose=1,backend="multiprocessing",timeout=60*15) as parallel:
        data = parallel(delayed(np.loadtxt)(file) for file in data_locs)
        pars = parallel(delayed(np.loadtxt)(file) for file in pars_locs)
    print("Successful loading")
    data = np.concatenate(data,axis=0)
    pars = np.concatenate(pars,axis=0)
    pars = pars[:,pars_list]
    pca_scaler = load("scalers/pca_scaler.bin")
    data = pca_scaler.transform(data)
    print("Successful transformation")
    train_data = data[:int(0.9*len(pars))]
    train_pars = pars[:int(0.9*len(pars))]
    val_data = data[int(0.9*len(pars)):]
    val_pars = pars[int(0.9*len(pars)):]
    pca = load("scalers/pca.bin")
    
    train_dataset = PCAtrimmedDataset(train_data,train_pars,
                                    pars_list,negatives,logged)
    val_dataset = PCAtrimmedDataset(val_data,val_pars,
                                      pars_list,negatives,logged)
    height_range = [np.log10(1.5),np.log10(7e2)]
    spin_range = [0,0.998]
    inclination_range = [np.log10(1),np.log10(89)]
    r_inner_range = [np.log10(1),np.log10(200)]
    r_outer_range = [np.log10(400),np.log10(1e5)]
    Gamma_range = [1.4,3.4]
    logxi_range = [0,4.7]
    Afe_range = [0.5,10]
    logNe_range = [15,20]
    kte_range = [np.log10(30),np.log10(500)]

    range_all = [height_range,spin_range,inclination_range,r_inner_range,
         r_outer_range,Gamma_range,logxi_range,Afe_range,
         logNe_range,kte_range]
    range_all = np.asarray(range_all)
    for i,rang in enumerate(range_all):
        print(f"pars out of range for parameter {i}:",
              np.any((np.array(val_dataset.pars[:,i])<rang[0])|(np.array(val_dataset.pars[:,i])>rang[1])))
    
    
    tra_loader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    
    loss_fn = PCALoss(pca.explained_variance_ratio_, device)
    comps = pca.n_components
    for i in range(1,10): 
        print("Training model")
        model = DynamicResNetwork(len(pars_list),comps,6,512)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        training_loop(model, optimizer, train, validate, tra_loader, 
              val_loader, loss_fn, device, f"rtfast_2_{i}", epochs = 2000)
    

if __name__ == "__main__":
    main()

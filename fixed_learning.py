"""
This is a quick training exercise to test if using PCA on spectra and using very
lightweight NNs is a viable alternative to what we've been doing up until now.
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from joblib import load

from dataStructures import PCAtrimmedDataset
from network import DynamicNetwork
from training import training_loop, train, validate, PCALoss

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pars_list = [0,1,2,3,4,6,7,8,9,10]
    negatives = [3]
    logged = [0,2,3,4,10]
    
    data_locs = [f"data/pca_comps/pca_comps_{i}.txt" for i in range(250)]
    pars_locs = [f"data/pars/pars_{i}.txt" for i in range(250)]
    
    pca = load("scalers/pca.bin")
    
    dataset = PCAtrimmedDataset(data_locs,pars_locs,pars_list,negatives,logged)
    train_dataset,val_dataset = torch.utils.data.random_split(dataset,
                                                              lengths=[0.9,0.1])
    val_loader = DataLoader(val_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    tra_loader = DataLoader(train_dataset, batch_size=1024, num_workers = 4, 
                                  shuffle=True)
    
    loss_fn = PCALoss(pca.explained_variance_ratio_, device)
    comps = pca.n_components
    
    print("Training model")
    model = DynamicNetwork(len(pars_list),comps,12,256)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    #optimizer = SGD(model.parameters(), lr=1e-4,momentum=0.9)
    training_loop(model, optimizer, train, validate, tra_loader, 
                  val_loader, loss_fn, device, "rtfast_2", epochs = 2000)
    

if __name__ == "__main__":
    main()

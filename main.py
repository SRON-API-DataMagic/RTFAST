"""
This is the main program that trains the neural network.
"""

import generator
import network
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
from torch.optim import Adam
import math
import time
import matplotlib.pyplot as plt

from sherpa.astro.ui import unpack_rmf
import os
import numpy as np

def create_blank_numpy_file():
    pars = [[6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,0,
            0,0.95,0,-0.8,0.3,2.2e-4,1,1.]]
    pars = np.asarray(pars)
    data = [np.zeros(4096)]
    data = np.asarray(data)
    np.save("data.npy",data)
    np.save("pars.npy",pars)

class custom_data(Dataset):
    
    def __init__(self,pars,data):
        super().__init__()
        self.par_list = pars
        self.data = data
    
    def __len__(self):
        return self.par_list.shape[0]
    
    def __getitem__(self,idx):
        datum = torch.from_numpy(self.data[idx])
        parameters = torch.from_numpy(self.par_list[idx])
        return datum,parameters

def pregenerate_models(n,egrid):
    all_data = []
    all_pars = []
    for i in range(n):
        print(f"Generating model {i+1}")
        pars = generator.pregen()
        data = generator.rtdist_flux(pars, egrid)
        all_data.append(data)
        all_pars.append(pars)
    all_data = np.asarray(all_data)
    all_pars = np.asarray(all_pars)
    new_pars = np.load("pars.npy")
    new_data = np.load("data.npy")
    with open("data.npy","wb") as f1:
        new_data = np.append(new_data,all_data,axis=0)
        np.save(f1,new_data)
    f1.close()
    with open("pars.npy","wb") as f2:
        new_pars = np.append(new_pars,all_pars,axis=0)
        np.save(f2,new_pars)
    f2.close()
    return all_data,all_pars

def save_data(data,pars):
    data = np.asarray(data)
    pars = np.asarray(pars)
    with open("data.npy","ab") as f:
        np.save(f,data)
    f.close()
    with open("pars.npy","ab") as f:
        np.save(f,pars)
    f.close()
    return
    

def train(dataloader,model,optimizer,loss_fn):
    model.train()
    size = len(dataloader.dataset)
    batch_size = 12
    loss_arr = 0
    for batch, (D,P) in enumerate(dataloader):
        D[D<1e-10] = 1e-10
        D = torch.log10(D)
        D = D.double()
        pred = model(P)
        loss = loss_fn(pred,D)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch*batch_size + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        loss_arr += loss
        batch += 1
    avg_loss = (loss_arr)/len(dataloader)
    return model, optimizer , avg_loss

def test(dataloader,model,loss_fn):
    model.eval()
    test_loss = 0
    batches = len(dataloader)
    with torch.no_grad():
        for batch, (D,P) in enumerate(dataloader):
            D[D<1e-10] = 1e-10
            D = torch.log10(D)
            pred = model(P)
            test_loss += loss_fn(pred, D).item()
    test_loss /= batches
    print(f"Avg loss: {test_loss:>8f}")
    return test_loss

torch.set_default_dtype(torch.double)

wrk_dir = os.getcwd()

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

print("Environmental variables successfully set")


rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"
rmf = unpack_rmf(rmf_name)
egrid = rmf.e_min

data,pars = pregenerate_models(100, egrid)
test_data = custom_data(pars, data)

with open("data.npy","rb") as f1:
    data = np.load(f1)

with open("pars.npy","rb") as f2:
    pars = np.load(f2)

print(np.all(data==0))
time.sleep(1)
data = custom_data(pars,data)


batch_size = 12
training_dataloader = DataLoader(data,batch_size = batch_size)
testing_dataloader = DataLoader(test_data,batch_size = batch_size)

model = network.NeuralNetwork(len(egrid))
optimizer = Adam(model.parameters())
max_iters = 10000
i = 0
imp = 0
loss_fn = nn.MSELoss()
loss = 100
last_sig_best = 1e7
tr_loss_arr = []
te_loss_arr = []
print("Beginning training")
while i < max_iters and imp < 50:
    print(f"Epoch {i+1} \n -----------------------")
    model, optimizer, train_loss = train(training_dataloader,model,optimizer,loss_fn)
    loss = test(testing_dataloader,model,loss_fn)
    te_loss_arr.append(loss)
    tr_loss_arr.append(train_loss)
    if train_loss > (last_sig_best - 0.1*last_sig_best):
        imp = 0
        last_sig_best = train_loss
    else:
        imp += 1
    i+=1
    
print("Completed training")
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

plt.plot(tr_loss_arr,label="training loss")
plt.plot(te_loss_arr,label="testing loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
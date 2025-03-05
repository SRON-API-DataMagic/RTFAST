"""
This is an experiment in using auto-encoders to emulate.
"""

from network import DynamicDecoder,DynamicAutoEncoder,DynamicEmulator
from network import DynamicNetwork, DynamicEncoder
from training import train_AE, test_AE, training_loop_AE
from joblib import load
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam
from dataStructures import DataStructure

def vae_gaussian_kl_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()

class CombinedLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.AELoss = nn.MSELoss()
        self.EmuLoss = nn.MSELoss()
        self.KL = nn.KLDivLoss()
    
    def forward(self,AE_pred,emu_pred,target):
        z,recon = AE_pred
        z_emu, recon_emu = emu_pred
        AELoss = self.AELoss(recon,target)
        EmuLoss = self.EmuLoss(z_emu,z)
        KLLoss = self.KL(z)
        return AELoss + EmuLoss + 0.1*KLLoss

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
    
    scaler = load("scalers/scaler.bin")
    
    data = scaler.transform(np.log10(spectra))
    
    split = int(0.9*len(data))
    
    batch_size = 1024
    assert len(pars) == len(data)
    
    train_data = DataStructure(pars[:split], data[:split])
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True)
    validation_data = DataStructure(pars[split:], data[split:])
    validation_loader = DataLoader(validation_data, batch_size=batch_size,
                                   shuffle=True)
    
    loss_fn = CombinedLoss()
    
    train = train_AE
    test =  test_AE
    num_pars=10
    spectrum_len = 999
    latent_space = 40
    nodes = 256
    num_layers = 4
    decoder = DynamicDecoder(spectrum_len, latent_space, nodes, num_layers)
    encoder = DynamicEncoder(spectrum_len, latent_space, nodes, num_layers)
    par_encoder = DynamicNetwork(num_pars, latent_space, num_layers=2, 
                                 nodes=128)
    AE = DynamicAutoEncoder(decoder, encoder)
    AE.to(device)
    AE.double()
    emulator = DynamicEmulator(decoder,par_encoder)
    emulator.to(device)
    emulator.double()
    optimizer = Adam(list(decoder.parameters()) + list(encoder.parameters())+ list(par_encoder.parameters()), 
                     lr=1e-4)
    #optimizer = SGD(model.parameters(), lr=1e-4,momentum=0.9)
    training_loop_AE(AE,emulator,optimizer,train,test,train_loader,
                  validation_loader, loss_fn, device, "reflection",
                  epochs = 200)
    
if __name__ == "__main__":
    main()
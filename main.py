"""
This is the main program that trains the neural network.
"""

import generator
import network
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sherpa.astro.ui import unpack_arf,unpack_rmf
import os

wrk_dir = os.getcwd()
rmf_name = wrk_dir+"/ResponseFiles/PN.rmf"

rmf = unpack_rmf(rmf_name)

egrid = rmf.e_min

pars = [6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,4e-5,20e-5,0.95,0,
        -0.8,0.3,2.2e-4,260000.,400,1,1.]

training_data = generator.rtdist_flux(pars, egrid)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size = batch_size)
model = network.NeuralNetwork()


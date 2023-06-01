"""
This program is built to purely test bottleneck performance in pytorch.
No modules in this file should be used in regular usage.
"""

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.preprocessing import MinMaxScaler

from network import NeuralNetwork
from main import maskedMSELoss, train, CustomData

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = NeuralNetwork(5, 4096)
model.to(device)

inp = torch.rand(1024,1,5).double().cuda()
mask = torch.randint(0,2,(1024,1,4096)).double().cuda()
data = torch.rand(1024,1,4096).double().cuda()

fname = "grid_5"
locations = "data/locations/"
scaler = MinMaxScaler()
num_workers = 4
batch_size = 1024

optimizer = Adam(model.parameters(),lr = 0.001)
loss_fn = maskedMSELoss

training_data = CustomData(locations+"loc_"+fname+".csv", scaler, 
                     scaler_name=f"{fname}_scaler.bin")

dataloader = DataLoader(training_data,batch_size=batch_size,
                              num_workers = num_workers, shuffle=True)

model(inp.to(device))

with profile(with_stack=True, profile_memory=True,
             activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], 
             record_shapes=True) as prof:
    with record_function("Model inference"):
        pred = model(inp.to(device))
    with record_function("Loss calculation"):
        loss = maskedMSELoss(pred, data.to(device), mask.to(device))
    with record_function("Training"):
        model, optimizer, train_loss = train(dataloader, model, optimizer, 
                                             loss_fn, device)

print(prof.key_averages(group_by_input_shape=True,group_by_stack_n=5).table(sort_by="cuda_memory_usage", 
                                                  row_limit=10))
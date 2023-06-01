"""
This program is built to purely test bottleneck performance in pytorch.
No modules in this file should be used in regular usage.
"""

import torch
import numpy as np
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from network import NeuralNetwork
from main import maskedMSELoss

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = NeuralNetwork(5, 4096)
model.to(device)

inp = torch.rand(1024, 5).double().cuda()
mask = torch.randint(0,2,(1024, 4096)).double().cuda()
data = torch.rand(1024,4096).double().cuda()

model(inp.to(device))

with profile(with_stack=True, profile_memory=True,
             activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], 
             record_shapes=True) as prof:
    with record_function("Model inference"):
        pred = model(inp.to(device))
    with record_function("Loss calculation"):
        loss = maskedMSELoss(pred, data.to(device), mask.to(device))

print(prof.key_averages(group_by_input_shape=True,group_by_stack_n=5).table(sort_by="cuda_time_total", 
                                                  row_limit=10))
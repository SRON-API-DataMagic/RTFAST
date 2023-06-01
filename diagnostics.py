"""
This program is built to purely test bottleneck performance in pytorch.
No modules in this file should be used in regular usage.
"""

import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler

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

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    pred = model(inp.to(device))
    loss = maskedMSELoss(pred, data.to(device), mask.to(device))

print(prof.key_averages().table(sort_by='self_cpu_time_total', 
                                                  row_limit=10))
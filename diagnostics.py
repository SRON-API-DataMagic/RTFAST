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

model = NeuralNetwork(5, 4096)

inp = torch.rand(128, 5).cuda()
mask = torch.randint(0,2,(128, 4096)).cuda()
data = torch.rand(128,4096).cuda()

model(inp)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    pred = model(inp)
    loss = maskedMSELoss(pred, data, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
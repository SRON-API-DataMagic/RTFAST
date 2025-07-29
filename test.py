import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from network import RTFAST

import os
import numpy as np
import f2py_interface as ib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from tqdm import tqdm

os.environ["REV_VERB" ] = "0"
os.environ["TEST_RUN" ] = "0"
os.environ["MU_ZONES" ] = "1"
os.environ["ION_ZONES"] = "20"
os.environ["A_DENSITY"] = "0"
os.environ["BACKSCL"  ] = "1.0"

rtfast = RTFAST()

Emin = 0.1
Emax = 100.0
ne = 1000
egrid = np.zeros(ne, dtype = np.float32)
for i in range(ne):
    egrid[i] = Emin * (Emax/Emin)**(i/ne)
    
pars = np.array([6,0.9,57,-1,2e4,0,2.45,3,1,15,50.,0,0,3e6,0,0,0,0,0,0,1,1.],dtype=np.float32)
test_array = ib.reltransDCp(egrid,pars)
prediction = rtfast(egrid,pars)

plt.plot(egrid[:-1],(egrid[:-1]**2)*test_array)
plt.plot(egrid[:-1],(egrid[:-1]**2)*prediction)
plt.loglog()
plt.xlabel("Energy (keV)")
plt.ylabel("EF(E)")
plt.savefig("figures/test.png")
plt.show()
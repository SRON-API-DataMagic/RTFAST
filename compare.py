"""
This program serves to visualise a neural network's outputs vs the true values.
"""
import torch
import generator
import network
import os
from sherpa.astro.ui import unpack_rmf
import matplotlib.pyplot as plt

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


model = network.NeuralNetwork(len(egrid))
model.load_state_dict(torch.load("model.pth"))
n = 10

for i in range(n):
    pars = generator.pregen()
    data = generator.rtdist_flux(pars, egrid)
    pred = model(pars)
    plt.plot(egrid,data,c="r",label="Truth")
    plt.plot(egrid,pred,c="blue",label="NN model")
    plt.legend()
    plt.xlabel("Energy in keV")
    plt.ylabel("Flux")
    plt.savefig(f"com_{i}.png")

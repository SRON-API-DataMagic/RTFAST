"""
This program deals with generating rtdist models with given parameters. For use
for generating sampels for training the emulator for rtdist.
"""

from reltrans import _models
import numpy as np
import torch

def rtdist_flux(pars,egrid):
    """
    

    Parameters
    ----------
    pars : array
        contains parameters used in simulation.
    egrid : array
        contains values of energy to calculate for.

    Returns
    -------
    model : array
        outputted simulated data.

    """
    model = _models.tdrtdist(pars,egrid)
    return model

def rtdist_lags(pars,egrid):
    """
    

    Parameters
    ----------
    pars : array
        contains parameters used in simulation.
    egrid : array
        contains values of energy to calculate for. Final point in array will
        always be zero when evaluated due to quirk in Sherpa/Xspec.

    Returns
    -------
    output : array
        outputted simulated data.

    """
    y = _models.tdrtdist(pars,egrid)
    dE = np.diff(egrid)
    output = y[:-1]/dE
    return output

def parameter_gen():
    height = np.linspace(1.3,1e4,(1e4 - 1.3)/0.1)
    spin = np.linspace(0,0.998,0.998/0.001)
    inc = np.linspace(1,80,79/1e-2)
    rin = np.linspace(-400,-1,399/1e-2)
    rout = np.linspace(400,1e5,(1e5 - 400))
    z = np.linspace(0,10,10/0.001)
    Gamma = np.linspace(1.4,3.4,2/0.01)
    Dkpc = np.logspace(np.log10(0.2),10,num=10000)
    Afe = np.linspace(0.5,10,9.5/1e-2)
    logNe = np.linspace(15,20,5/0.01)
    nH = np.logspace(-10,6,num=10000)
    boost = np.linspace(1e-2,10,10/1e-2)
    mass = np.logspace(0,11,num=10000)
    honr = np.linspace(0,0.176,0.176/1e-4)
    b1 = np.linspace(0,2,2/1e-2)
    b2 = np.linspace(-4,4,8/1e-2)
    fmin = np.logspace(-10,6,num=10000)
    fmax = np.logspace(-10,6,num=10000)
    phiAB = np.linspace(-6.283,6.283,(6.283)*2/0.001)
    g = np.linspace(0,0.5,0.5/0.01)
    Anorm = np.logspace(-12,10,num=10000)
    
    grid = np.meshgrid(height,spin,inc,rin,rout,z,Gamma,Dkpc,Afe,logNe,nH,boost,
                       mass,honr,b1,b2,fmin,fmax,phiAB,g,Anorm)
    print(grid)
    return grid


def lag_gen(egrid):
    pars = parameter_gen()
    data = []
    for i,par in enumerate(pars):
        return
    return data

parameter_gen()
"""
This program deals with generating rtdist models with given parameters. For use
for generating sampels for training the emulator for rtdist.
"""
import numpy as np
from reltrans import _models

def par_gen():
    uni = np.random.uniform
    pars = [6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,4e-5,20e-5,0.95,0,
            -0.8,0.3,2.2e-4,260000.,400,1,1.]
    
    h = 10**(uni(np.log10(4),np.log10(300)))
    a = uni(0.1,0.998)
    inc = uni(1,80)
    rin = uni(1,300)
    rout = uni(400,1e5)
    z = uni(0,0.1)
    Gamma = uni(1,4)
    Dkpc = uni(0.2,1e10)
    Afe = uni(0.5,10.)
    logNe = uni(0,1e6)
    boost = uni(1e-2,10)
    mass = uni(1,1e11)
    honr = uni(0,0.15)
    b1 = uni(0,2)
    b2 = uni(-4,4)
    fmin = 4e-5
    fmax = 20e-5
    phiA = 0
    phiAB = uni(-6.283,6.283)
    g = uni(0,0.5)
    Anorm = 1
    RESP = 1
    pars = [h,a,inc,rin,rout,z,Gamma,Dkpc,Afe,logNe,boost,mass,honr,b1,b2,fmin,
            fmax,phiA,phiAB,g,Anorm,RESP]
    
    return pars


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


def pregen():
    pars = [6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,4e-5,20e-5,0.95,0,
            -0.8,0.3,2.2e-4,260000.,400,1,1.]
    uni = np.random.uniform
    a = uni(0.1,0.998)
    mass = 10**uni(1,11)
    
    pars[1] = a
    pars[13] = mass
    
    return pars

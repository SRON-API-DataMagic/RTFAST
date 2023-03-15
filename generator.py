"""
This program deals with generating rtdist models with given parameters. For use
for generating sampels for training the emulator for rtdist.
"""
import numpy as np
from reltrans import _models

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
    """
    Generates random parameters for the rtdist model to evaluate

    Returns
    -------
    pars : list
        list of parameters for evaluation by rtdist.

    """
    pars = [6,0.9,57,-1,2e4,0.024917,2.45,1e5,1,17,50.,5,1,3e6,0.02,0,0,0,0,0.95,0,
            -0.8,0.3,2.2e-4,1,1.]
    uni = np.random.uniform
    a = uni(0.1,0.998)
    mass = 10**uni(1,11)
    
    pars[1] = a
    pars[13] = mass
    
    return pars

"""
This program deals with generating rtdist models with given parameters. For use
for generating sampels for training the emulator for rtdist.
"""
import numpy as np
from reltrans import _models
from torch import data

class IterDataset(data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()

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

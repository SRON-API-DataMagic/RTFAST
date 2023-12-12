"""
This program simulates a test case and attempts a simple MCMC fit using the
emcee package and the emulator.
"""

import emcee
from reltrans import _models

import torch

def emulator(theta):
    """
    The function calls the emulator and returns the predicted spectra.

    Parameters
    ----------
    theta : np.ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """
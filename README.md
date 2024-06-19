This repository contains the code necessary to train a neural network emulator for the rtdist 
AGN/black hole spectral model. It is built with pytorch. We achieved an order of 1% error across
the entirety of the spectrum and parameter space for a 20 free parameter model.

The emulator trained with this code (RTFAST) greatly reduce the computation time, allowing us to
calculate bayesian posteriors for these relatively high dimensional problems. The finished 
emulator can be found at this public repository for public use.

We utilised latin hyper cube sampling as well as PCA decomposition to simplify the constraints of
the original problem to a realistic, extremely lightweight, fast running drop in emulator.

We note and encourage that this code can be used as a simple starting template for those with 
an x-ray spectral model from xspec that they wish to build an emulator for. Please contact
Benjamin Ricketts (the author of this repository) if you wish to know more or would like guidance
with your particular project.

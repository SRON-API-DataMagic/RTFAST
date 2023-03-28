This program is built to train a neural network emulator for the rtdist AGN/black hole model. 
It is built with pytorch. We aim to obtain a 1% error in flux and time lag outputs due to 
comparable systemic uncertainties. This emulator should greatly reduce the computation time
associated with calculating bayesian uncertainties for these high dimensional problems.

We utilise latin hyper cube sampling and active learning so as to more efficiently learn the
very large parameter space that rtdist explores. In terms of active learning, we utilize the
query-by-dropout-committee method outlined by Constraining the Parameters of High-Dimensional
Models with Active Learning (Caron et al 2019).

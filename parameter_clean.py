import f2py_interface as ib
import numpy as np
import os
from joblib import load
import re

def is_aligned(truth,result):
    #checks if values are within 1% of each other
    aligned = np.all(np.where((truth-result)/truth<0.01)) 
    return aligned

def find_misalignment(parameters,results):
    left = 0
    right = len(results)  # Since results is shorter

    while left < right:
        mid = (left + right) // 2
        truth = ib.reltransDCp(egrid, parameters[mid])
        if is_aligned(truth, results[mid]):
            left = mid + 1
        else:
            right = mid

    return left  # This is the index in parameters with no matching result

Emin = 0.1
Emax = 100.0
ne = 1000
egrid = np.zeros(ne, dtype = np.float32)
for i in range(ne):
    egrid[i] = Emin * (Emax/Emin)**(i/ne)

os.environ["REV_VERB" ] = "0"
os.environ["MU_ZONES" ] = "1"
os.environ["ION_ZONES"] = "1"
os.environ["A_DENSITY"] = "0"
os.environ["BACKSCL"  ] = "1.0"
os.environ["TEST_RUN" ] = "0"

size = 250
data_locs = [f"data/pca_comps/pca_comps_{i}.txt" for i in range(size)]
pars_locs = [f"data/pars/pars_{i}.txt" for i in range(size)]

scaler = load("scalers/scaler.bin")
pca = load("scalers/pca.bin")

for par_loc,data_loc in zip(data_locs,pars_locs):
    number_str = re.sub(r'\D', '', par_loc)
    parameters = np.loadtxt(par_loc)
    data = np.loadtxt(data_loc)
    results = 10**scaler.inverse_transform(pca.inverse_transform(data))
    counter = 0
    while parameters.shape[0] != data.shape[0]:
        misaligned_index = find_misalignment(parameters, results)
        parameters.pop(misaligned_index)
        counter += 1
    print("Removed",counter,"parameters")
    truth = ib.reltransDCp(egrid, parameters[-1])
    if is_aligned(truth, results[-1]):
        print("final model is aligned")
        np.savetxt("data/pca_comps/pca_comps_{i}_clean.txt")
    else:
        print("failed to align parameters with model results")
        

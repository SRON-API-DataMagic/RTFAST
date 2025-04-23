import glob
import f2py_interface as ib
import numpy as np
import os
from joblib import load
import re

def is_aligned(truth,result):
    #checks if values are within 1% of each other
    aligned = np.all((truth-result)/truth<0.1)
    return aligned

def find_misalignment(parameters,results):
    left = 0
    right = len(results)  # Since results is shorter

    while left < right:
        mid = (left + right) // 2
        print(mid,":",parameters[mid])
        truth = ib.reltransDCp(egrid, parameters[mid])
        if is_aligned(truth, results[mid]):
            left = mid + 1
        else:
            right = mid
    print("Found misalignment",left)
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

data_locs = sorted(glob.glob("data/pca_comps/pca_comps_*.txt"))
pars_locs = sorted(glob.glob("data/pars/pars_*.txt"))

scaler = load("scalers/scaler.bin")
pca = load("scalers/pca.bin")

for data_loc,par_loc in zip(data_locs,pars_locs):
    number_str = re.sub(r'\D', '', par_loc)
    parameters = np.loadtxt(par_loc).astype(np.float32)
    data = np.loadtxt(data_loc)
    results = 10**scaler.inverse_transform(pca.inverse_transform(data))
    test = ib.reltransDCp(egrid, parameters[-1])
    misalignment = is_aligned(ib.reltransDCp(egrid, parameters[-1]), results[-1])
    if ~misalignment:
        counter = 0
        while ~misalignment:
            misaligned_index = find_misalignment(parameters, results)
            parameters = np.delete(parameters,misaligned_index,axis=0)
            counter += 1
            misalignment = is_aligned(ib.reltransDCp(egrid, parameters[-1]), results[-1])
        print("Removed",counter,"parameters")
        truth = ib.reltransDCp(egrid, parameters[-1])
        success = is_aligned(truth, results[-1])
        if is_aligned(truth, results[-1]):
            print("final model is aligned")
            np.savetxt(f"data/pars/pars_{number_str}_clean.txt",parameters)
        else:
            print("failed to align parameters with model results")
    else:
        print(f"data {number_str} is not misaligned")
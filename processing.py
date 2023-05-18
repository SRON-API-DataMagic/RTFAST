"""
This module generally deals with data processing for plotting and retrieval
"""

import numpy as np
import os, glob

def breakup(dataset_loc,pars_loc,destination):
    dataset = np.loadtxt(dataset_loc)
    pars = np.loadtxt(pars_loc)
    
    locations = []
    for i,(spectra,pars) in enumerate(zip(dataset,pars)):
        loc = f"data/spectra/spectra_{i}.txt"
        print(loc)
        print(spectra)
        np.savetxt(loc,spectra)
        row = pars.tolist()+[loc]
        locations.append(row)
    
    locations = np.asarray(locations)
    np.savetxt("data/spectra_loc.txt",locations)
    

def main():
    dataset_loc = "data/loop_30_data.txt"
    pars_loc = "data/loop_30_pars.txt"
    destination = "data/spectra"
    breakup(dataset_loc, pars_loc, destination)

if __name__ == "__main__":
    main()
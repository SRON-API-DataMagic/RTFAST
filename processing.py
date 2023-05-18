"""
This module generally deals with data processing for plotting and retrieval
"""

import numpy as np

def breakup(dataset_loc,pars_loc,destination):
    dataset = np.loadtxt(dataset_loc)
    
    locations = []
    for i,spectra in enumerate(dataset):
        loc = f"data/spectra/spectra_{i}.txt"
        print(loc)
        np.savetxt(loc,spectra)
        locations.append(loc)
    
    locations = np.asarray(locations)
    np.savetxt("data/locations/loc_25.txt",locations, fmt='%s')
    

def main():
    dataset_loc = "data/loop_25_data.txt"
    pars_loc = "data/loop_25_pars.txt"
    destination = "data/spectra"
    breakup(dataset_loc, pars_loc, destination)

if __name__ == "__main__":
    main()
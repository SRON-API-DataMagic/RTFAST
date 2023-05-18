"""
This module generally deals with data processing for plotting and retrieval
"""

import numpy as np
import tqdm

def breakup(dataset_loc,pars_loc,destination,index):
    dataset = np.loadtxt(dataset_loc)
    
    locations = []
    for i,spectra in enumerate(tqdm.tqdm(dataset)):
        loc = f"data/spectra/spectra_{i}.txt"
        #np.savetxt(loc,spectra)
        locations.append(loc)
    
    locations = np.asarray(locations)
    np.savetxt(f"data/locations/loc_{index}.txt",locations, fmt='%s')
    

def main():
    destination = "data/spectra"
    indexes = [0,1,2,3,4,5,10,15]
    for index in indexes:
        print("Converting loop",index)
        dataset_loc = f"data/loop_{index}_data.txt"
        pars_loc = f"data/loop_{index}_pars.txt"
        breakup(dataset_loc, pars_loc, destination,index)

if __name__ == "__main__":
    main()
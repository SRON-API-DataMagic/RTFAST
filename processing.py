"""
This module generally deals with data processing for plotting and retrieval
"""

import numpy as np
import tqdm
import glob

def breakup(dataset_loc,pars_loc,destination,index):
    dataset = np.loadtxt(dataset_loc)
    
    files = glob.glob("./data/spectra/*.txt")
    for i,file in enumerate(files):
        tmp = file.replace("./data/spectra/spectra_","")
        tmp = int(tmp.replace(".txt",""))
        files[i] = tmp
    files = np.asarray(files)
    start = files.max() + 1
    
    locations = []
    for i,spectra in enumerate(tqdm.tqdm(dataset),start=start):
        loc = f"data/spectra/spectra_{i}.txt"
        np.savetxt(loc,spectra)
        locations.append(loc)
    
    locations = np.asarray(locations)
    np.savetxt(f"data/locations/grid_loc_{index}.txt",locations, fmt='%s')
    

def main():
    destination = "data/spectra"
    indexes = [10]
    for index in indexes:
        print("Converting grid",index)
        dataset_loc = f"data/grid_{index}_data.txt"
        pars_loc = f"data/grid_{index}_pars.txt"
        breakup(dataset_loc, pars_loc, destination,index)

if __name__ == "__main__":
    main()
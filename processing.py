"""
This module generally deals with data processing for plotting and retrieval
"""

import numpy as np
import tqdm
import glob
import pandas as pd

def breakup(dataset_loc,pars_loc,destination):
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
    np.savetxt("data/locations/test_loc.txt",locations, fmt='%s')
    return

def merging_locations_pars(pars_loc,locations_loc,destination):
    pars = np.loadtxt(pars_loc)
    locations = np.loadtxt(locations_loc,dtype='%s')
    column_names = ["h","a","inc","rin","rout","z","Gamma","Dkpc","Afe","logNe",
                    "kTe","nH","boost","Mass","honr","b1","b2","fmin","fmax",
                    "ReIm","phiA","phiAB","g","Anorm","RESP","Xnorm"]
    pars_df = pd.DataFrame(pars,columns = column_names)
    print(pars_df)
    locations_df = pd.DataFrame(locations,columns=["Location"])
    print(locations_df)
    final_df = pd.concat([pars_df,locations_df],axis = 1, join = "inner")
    print(final_df)
    return
    

def main():
    destination = "data/"
    locations_loc ="data/locations/loc_"
    pars_loc = "data/pars/"
    
    indexes = ["0","1","2","3","4","5","10","15","20","25","grid_5","grid_6",
               "grid_7","grid_8","grid_9","grid_10"]
    
    for index in indexes:
        loc_index = locations_loc+index+".txt"
        pars_index = pars_loc+index+"_pars.txt"
        merging_locations_pars(pars_index, loc_index, destination)
    
if __name__ == "__main__":
    main()
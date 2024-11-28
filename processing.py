"""
This module generally deals with data processing for plotting, retrieval and
saving data.
"""

import numpy as np
import glob
import pandas as pd
import os
from joblib import Parallel, delayed
from pathlib import Path

def save_file(i,item,destination):
    loc = f"{destination}/spectra_{i}.txt"
    np.savetxt(loc,item)
    return loc

def saveData(dataset, pars, locs_destination, data_destination):
    """
    Checks for existing data on disk and saves data in individual units to disk 
    that doesn't intefere with existing data. Saves the parameters associated 
    with the data to a seperate file with the location of the corresponding 
    data.

    Parameters
    ----------
    dataset : ndarray
        array that contains data to be saved (e.g. spectra or time lags).
    pars : ndarray
        array that contains corresponding parameters used to generate the data
        being saved.
    destination : string
        folder for locations of data to be saved.
    fname : string
        name of file that locations will be saved in.
    current_locs : string, optional
        location of locations to add to if desired. The default is None.
    lags : string, optional
        if this is anything but None, will assume that the data being worked
        with is of time lags format. The default is None.

    Returns
    -------
    None.

    """
    #find the last spectra's name and use that to save new spectra without
    #overwriting
    try:
        files = glob.glob(f"{data_destination}/*.txt")
        for i,file in enumerate(files):
            tmp = file.replace(f"{data_destination}/spectra_","")
            tmp = int(tmp.replace(".txt",""))
            files[i] = tmp
        files = np.asarray(files)
        start = files.max() + 1
    except:
        start = 0
    
    cpu_num = os.cpu_count()
    #save data to disk and save location to dataset
    locations = Parallel(n_jobs=cpu_num,verbose=1)(delayed(save_file)(i,item,data_destination) for i,item in enumerate(dataset,start=start))
    
    locations = np.asarray(locations)
    column_names = ["h","a","inc","rin","rout","z","Gamma","Dkpc","Afe","logNe",
                    "kTe","nH","boost","Mass","honr","b1","b2","fmin","fmax",
                    "ReIm","phiA","phiAB","g","Anorm","RESP","Xnorm"]
    pars_df = pd.DataFrame(pars,columns = column_names)
    locations_df = pd.DataFrame(locations,columns=["Location"])
    df = pd.concat([pars_df,locations_df],axis = 1, join = "inner")
    
    my_file = Path(locs_destination)
    if my_file.is_file():
        current_locs = pd.read_csv(locs_destination)
        df = pd.concat([df,current_locs],axis=0,ignore_index=True)
    
    df.to_csv(locs_destination,index=False)
    return

def spectraChecker(flux,pars_flux,threshold):
    """
    

    Parameters
    ----------
    flux_loc : string
        location of flux spectra table.
    lags_loc : ndarray
        location of lags table.
    upper_threshold: float
        number of photons/s/cm^2 that 25% or less of the spectra should be under
    lower_threshold: float
        number of photons/s/cm^2 that 25% or less of the spectra should be over
    Returns
    -------
    indexes: list
        list of indices of parameters to be removed

    """
    indexes = []
    for i, spec in enumerate(flux):
        if np.any(spec > threshold) != True:
            indexes.append(i)
        elif np.any(spec<0) == True:
            indexes.append(i)
    if indexes != []:
        print(f"A total of {len(indexes)} spectra were ineligible.")
    else:
        print(f"There were no uneligible spectra: {indexes}")
    index = np.unique(indexes).tolist()
    flux = np.delete(flux,index,axis=0)
    pars_flux = np.delete(pars_flux,index,axis=0)
    return flux, pars_flux

def readAndRemoveNans(flux_loc):
    flux_df = pd.read_csv(flux_loc)
    index = []
    for i,row in flux_df.iterrows():
        spec = np.loadtxt(row["Location"])
        if np.any(np.isnan(spec)) == True or np.any(np.isinf(spec)):
            index.append(i)
    
    try:
        index = np.unique(index).tolist()
        print("Found bad models:", index)
    except:
        print("No bad models found")
        index = []
    if index != []:
        print("Found bad models, printing parameters...")
        for indice in index:
            print(f"{indice}: {flux_df.iloc[indice]}")
    else:
        print("No bad models")
    flux_df.drop(index,inplace=True)
    flux_df.to_csv(flux_loc, index=False)
    return

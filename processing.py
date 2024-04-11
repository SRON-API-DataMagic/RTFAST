"""
This module generally deals with data processing for plotting, retrieval and
saving data.
"""

import numpy as np
import tqdm
import glob
import pandas as pd
import torch
import os
from joblib import Parallel, delayed

def mergeSaveData(new_data,old_data,destination,fname):
    """
    Method that takes two existing groups of data and collates them together.
    Usually used for combining training and validation data after an active
    learning loop.

    Parameters
    ----------
    new_data : string
        location of new data.
    old_data : string
        location of old data.
    destination : string
        folder to save new file in.
    fname : string
        new filename of the collated data.

    Returns
    -------
    None.

    """
    df = pd.concat([old_data,new_data],axis=0,ignore_index=True)
    df.to_csv(destination+fname,index=False)
    return

def saveData(dataset, pars, destination, fname, current_locs = None, lags = None):
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
        if lags == None:
            files = glob.glob("./data/spectra/*.txt")
            for i,file in enumerate(files):
                tmp = file.replace("./data/spectra/spectra_","")
                tmp = int(tmp.replace(".txt",""))
                files[i] = tmp
        else:
            files = glob.glob("./data/lags/*.txt")
            for i,file in enumerate(files):
                tmp = file.replace("./data/lags/lags_","")
                tmp = int(tmp.replace(".txt",""))
                files[i] = tmp
        files = np.asarray(files)
        start = files.max() + 1
    except:
        start = 0
    
    def save_file(i,item,lags):
        if lags == None:
            loc = f"data/spectra/spectra_{i}.txt"
        else:
            loc = f"data/lags/lags_{i}.txt"
        np.savetxt(loc,item)
        return loc
    
    cpu_num = os.cpu_count()
    #save data to disk and save location to dataset
    locations = Parallel(n_jobs=cpu_num,verbose=1)(delayed(save_file)(i,item,lags) for i,item in enumerate(dataset,start=start))
    
    locations = np.asarray(locations)
    column_names = ["h","a","inc","rin","rout","z","Gamma","Dkpc","Afe","logNe",
                    "kTe","nH","boost","Mass","honr","b1","b2","fmin","fmax",
                    "ReIm","phiA","phiAB","g","Anorm","RESP","Xnorm"]
    pars_df = pd.DataFrame(pars,columns = column_names)
    locations_df = pd.DataFrame(locations,columns=["Location"])
    df = pd.concat([pars_df,locations_df],axis = 1, join = "inner")
    
    if current_locs is not None:
        df = pd.concat([df,current_locs],axis=0,ignore_index=True)
    
    df.to_csv(destination+fname,index=False)
    return

def removeParameters(df_loc,indexes):
    df = pd.read_csv(df_loc)
    df.drop(indexes,inplace=True)
    df.to_csv(df_loc,index=False)
    return

def removeRedundantData():
    """
    Method that checks if all data currently saved on disk exists as referenced
    by the list of locations and if it doesn't exist, deletes it.

    Returns
    -------
    None.

    """
    locations = "./data/locations/"
    labels = ["locs_10_spectra_tra.csv","locs_10_spectra_val.csv",
              "locs_20_spectra_tra.csv","locs_20_spectra_val.csv",
              "loc_flux_test.csv"]
    
    spectra_names = []
    for fname in labels:
        data = pd.read_csv(locations+fname)
        print(f"{fname}:{len(data)}")
        names = data["Location"].values.tolist()
        spectra_names.extend(names)
    
    spectra_names = np.asarray(spectra_names)
    files = glob.glob("data/spectra/*.txt")
    files = np.asarray(files)
    diff = np.setdiff1d(files,spectra_names)
    print(f"Removing {len(diff)} files...")
    for file in tqdm.tqdm(diff):
        os.remove(file)
    
    return

def renameData(df,destination,fname):
    """
    Changes the name of a locations file as well as moving it to a new folder
    if wished.

    Parameters
    ----------
    data_locs : string
        location of file.
    destination : string
        new folder to be saved to.
    fname : string
        new name of file.

    Returns
    -------
    None.

    """
    df.to_csv(destination+fname,index=False)
    return

def loadData(location):
    """
    Reads in panda csv file

    Parameters
    ----------
    location : string
        location of file.

    Returns
    -------
    pandas dataframe
        returns pandas dataframe that is read.

    """
    return pd.read_csv(location)

def saveLoop(model,data_locs,optimizer,te_loss,tr_loss,num,epochs,typ="flux"):
    """
    Saves information and model for an active learning loop to disk for future
    use.

    Parameters
    ----------
    model : pytorch model
        the current version of the model to be saved to disk.
    data_locs : string
        location of the currect data being used to train the network.
    optimizer : pytorch optimizer
        the current version of the optimizer used to train the network.
    te_loss : ndarray
        average testing loss for each epoch of the network trained so far.
    tr_loss : ndarray
        average training loss for each epoch of the network trained so far.
    num : int
        the current active learning loop number.
    epochs : ndarray
        the epoch that each active learning loop stopped at so far.
    typ : string, optional
        signifies which type of network is being trained and is then saved into
        filenames for identification later on. The default is "flux".

    Returns
    -------
    None.

    """
    print("Saving loop")
    renameData(pd.read_csv(data_locs),destination = "data/locations/",
              fname = f"loc_{typ}_{num}.csv")
    print("Saved data")
    torch.save(model.state_dict(), f"models/{num}_{typ}_model.pth")
    print("Saved model")
    torch.save(optimizer.state_dict(),f"models/{num}_{typ}_optimizer.pth")
    print("Saved optimizer")
    np.savetxt(f"loss/{num}_{typ}_te_loss.txt",te_loss)
    np.savetxt(f"loss/{num}_{typ}_tr_loss.txt",tr_loss)
    np.savetxt(f"loss/{num}_{typ}_epochs.txt",epochs)
    print("Saved losses")
    return

def nanChecker(data,pars):
    """
    Returns the row indices of any data that contains NaNs for removal

    Parameters
    ----------
    data : ndarray
        array of data to be checked.
    pars : ndarray
        array of associated parameters. Is printed if the associated model has
        NaN values.

    Returns
    -------
    index : list
        list of indices of data to be removed.

    """
    index = []
    for i,spec in enumerate(data):
        if np.any(np.isnan(spec)) == True or np.any(np.isinf(spec)):
            index.append(i)
    if index != []:
        print("Found bad models, printing parameters...")
        for indice in index:
            print(f"{indice}: {pars[indice]}")
    return index

def spectraChecker(flux,pars_flux,pars_lags,threshold):
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
    pars_lags = np.delete(pars_lags,index,axis=0)
    return flux, pars_flux, pars_lags

def readAndRemoveNans(flux_loc,lags_loc):
    flux_df = pd.read_csv(flux_loc)
    lags_df = pd.read_csv(lags_loc)
    index = []
    for i,row in flux_df.iterrows():
        spec = np.loadtxt(row["Location"])
        if np.any(np.isnan(spec)) == True or np.any(np.isinf(spec)):
            index.append(i)
    for i,row in lags_df.iterrows():
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
    lags_df.drop(index,inplace=True)
    lags_df.to_csv(lags_loc, index=False)
    return

def main():
    removeRedundantData()
    
if __name__ == "__main__":
    main()
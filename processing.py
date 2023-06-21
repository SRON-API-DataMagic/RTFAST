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

def mergeSaveData(new_data,old_data,destination,fname):
    df = pd.concat([old_data,new_data],axis=0,ignore_index=True)
    df.to_csv(destination+fname,index=False)
    return

def saveData(dataset, pars, destination, fname, current_locs = None):
    
    try:
        files = glob.glob("./data/spectra/*.txt")
        for i,file in enumerate(files):
            tmp = file.replace("./data/spectra/spectra_","")
            tmp = int(tmp.replace(".txt",""))
            files[i] = tmp
        files = np.asarray(files)
        start = files.max() + 1
    except:
        start = 0
    
    locations = []
    for i,spectra in enumerate(tqdm.tqdm(dataset),start=start):
        loc = f"data/spectra/spectra_{i}.txt"
        np.savetxt(loc,spectra)
        locations.append(loc)
    
    locations = np.asarray(locations)
    column_names = ["h","a","inc","rin","rout","z","Gamma","Dkpc","Afe","logNe",
                    "kTe","nH","boost","Mass","honr","b1","b2","fmin","fmax",
                    "ReIm","phiA","phiAB","g","Anorm","RESP","Xnorm"]
    pars_df = pd.DataFrame(pars,columns = column_names)
    locations_df = pd.DataFrame(locations,columns=["Location"])
    df = pd.concat([pars_df,locations_df],axis = 1, join = "inner",ignore_index=True)
    
    if current_locs is not None:
        df = pd.concat([df,current_locs],axis=0,ignore_index=True)
    
    df.to_csv(destination+fname,index=False)
    return

def removeRedundantData():
    locations = "./data/locations/"
    labels = ["loc_60.csv","loc_test.csv","loc_30_bar.csv",
              "loc_grid_5.csv","loc_grid_5_test.csv",
              "loc_grid_6.csv","loc_grid_6_test.csv",
              "loc_grid_7.csv","loc_grid_7_test.csv",
              "loc_grid_8.csv","loc_grid_8_test.csv",
              "loc_grid_9.csv","loc_grid_9_test.csv",
              "loc_grid_10.csv","loc_grid_10_test.csv"]
    
    spectra_names = []
    for fname in labels:
        data = pd.read_csv(locations+fname)
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

def renameData(data_locs,destination,fname):
    df = pd.read_csv(data_locs)
    df.to_csv(destination+fname,index=False)
    return

def loadData(location):
    return pd.read_csv(location)

def merging_locations_pars(pars_loc, locations_loc, destination,index):
    pars = np.loadtxt(pars_loc)
    locations = np.loadtxt(locations_loc,dtype=str)
    column_names = ["h","a","inc","rin","rout","z","Gamma","Dkpc","Afe","logNe",
                    "kTe","nH","boost","Mass","honr","b1","b2","fmin","fmax",
                    "ReIm","phiA","phiAB","g","Anorm","RESP","Xnorm"]
    pars_df = pd.DataFrame(pars,columns = column_names)
    locations_df = pd.DataFrame(locations,columns=["Location"])
    final_df = pd.concat([pars_df,locations_df],axis = 1, join = "inner",ignore_index=True)
    final_df.to_csv(destination+"loc_"+index+".csv",index=False)
    return

def loadSpectra(df,egrid):
    spectra = np.zeros((len(df),len(egrid)))
    for index, row in df.iterrows():
        spectra[index]=np.loadtxt(row["Location"])
    return spectra

def loadParameters(df):
    parameters = np.zeros((len(df),len(df.columns)-2))
    for index, row in df.iterrows():
        parameters[index]=row.iloc[0:len(df.columns)-1]
    return parameters

def saveLoop(model,data_locs,optimizer,te_loss,tr_loss,num,epochs):
    print("Saving loop")
    renameData(data_locs,destination = "data/locations/",
              fname = f"loc_{num}.csv")
    print("Saved data")
    torch.save(model.state_dict(), f"models/{num}_model.pth")
    print("Saved model")
    torch.save(optimizer.state_dict(),f"models/{num}_optimizer.pth")
    print("Saved optimizer")
    np.savetxt(f"loss/{num}_te_loss.txt",te_loss)
    np.savetxt(f"loss/{num}_tr_loss.txt",tr_loss)
    np.savetxt(f"loss/{num}_epochs.txt",epochs)
    print("Saved losses")
    return

def nanChecker(data,pars):
    index = []
    for i,spec in enumerate(data):
        if np.any(np.isnan(spec)) == True:
            index.append(i)
    if index != []:
        print("Found bad models, printing parameters...")
        for indice in index:
            print(f"{indice}: {pars[indice]}")
        data = np.delete(data,index, axis=0)
        pars = np.delete(pars,index, axis=0)
    return data, pars

def main():
    removeRedundantData()
    
if __name__ == "__main__":
    main()
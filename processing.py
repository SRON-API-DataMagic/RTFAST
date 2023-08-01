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

def saveData(dataset, pars, destination, fname, current_locs = None, lags = None):
    
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
    
    locations = []
    for i,item in enumerate(tqdm.tqdm(dataset),start=start):
        if lags == None:
            loc = f"data/spectra/spectra_{i}.txt"
        else:
            loc = f"data/lags/lags_{i}.txt"
        np.savetxt(loc,item)
        locations.append(loc)
    
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

def removeRedundantData():
    locations = "./data/locations/"
    labels = ["loc_flux_30.csv","loc_flux_test.csv",
              "loc_lags_30.csv","loc_lags_test.csv",
              "loc_grid_5_lags.csv","loc_grid_5_lags_test.csv",
              "loc_grid_6_lags.csv","loc_grid_6_lags_test.csv",
              "loc_grid_7_lags.csv","loc_grid_7_lags_test.csv",
              "loc_grid_8_lags.csv","loc_grid_8_lags_test.csv",
              "loc_grid_9_lags.csv","loc_grid_9_lags_test.csv",
              "loc_grid_10_lags.csv","loc_grid_10_lags_test.csv",
              "loc_grid_5_flux.csv","loc_grid_5_flux_test.csv",
              "loc_grid_6_flux.csv","loc_grid_6_flux_test.csv",
              "loc_grid_7_flux.csv","loc_grid_7_flux_test.csv",
              "loc_grid_8_flux.csv","loc_grid_8_flux_test.csv",
              "loc_grid_9_flux.csv","loc_grid_9_flux_test.csv",
              "loc_grid_10_flux.csv","loc_grid_10_flux_test.csv"]
    
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

def saveLoop(model,data_locs,optimizer,te_loss,tr_loss,num,epochs,typ="flux"):
    print("Saving loop")
    renameData(data_locs,destination = "data/locations/",
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
    index = []
    for i,spec in enumerate(data):
        if np.any(np.isnan(spec)) == True or np.any(np.isinf(spec)):
            index.append(i)
    if index != []:
        print("Found bad models, printing parameters...")
        for indice in index:
            print(f"{indice}: {pars[indice]}")
    return index

def main():
    removeRedundantData()
    
if __name__ == "__main__":
    main()
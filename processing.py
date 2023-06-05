"""
This module generally deals with data processing for plotting, retrieval and
saving data.
"""

import numpy as np
import tqdm
import glob
import pandas as pd
import torch

def mergeSaveData(new_data,old_data,destination,fname):
    df = pd.concat([old_data,new_data],axis=0,ignore_index=True)
    df.to_csv(destination+fname)
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
    df = pd.concat([pars_df,locations_df],axis = 1, join = "inner")
    
    if current_locs is not None:
        df = pd.concat([df,current_locs],axis=0,ignore_index=True)
    
    df.to_csv(destination+fname)
    return

def renameData(data_locs,destination,fname):
    df = pd.read_csv(data_locs)
    df.to_csv(destination+fname)
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
    final_df = pd.concat([pars_df,locations_df],axis = 1, join = "inner")
    final_df.to_csv(destination+"loc_"+index+".csv")
    return

def saveLoop(model,data_locs,optimizer,te_loss,tr_loss,num,epochs):
    print("Saving loop")
    renameData(data_locs,destination = "data/locations/",
              fname = f"loc_{num}.csv")
    print("Saved data")
    torch.save(model.state_dict(), f"models/{num}_model.pth")
    print("Saved model")
    torch.save(optimizer.state_dict())
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
    destination = "data/locations/"
    locations_loc ="data/locations/loc_"
    pars_loc = "data/pars/"
    
    indexes = ["test"]
    
    for index in indexes:
        print(index)
        loc_index = locations_loc+index+".txt"
        pars_index = pars_loc+index+"_pars.txt"
        merging_locations_pars(pars_index, loc_index, destination, index)
        print(pd.read_csv("data/locations/loc_"+index+".csv"))
    
if __name__ == "__main__":
    main()
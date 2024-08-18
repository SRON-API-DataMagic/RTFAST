"""
This program loads and scales spectra into a series of PCA components for 
training. Loading PCA components directly after scaling parameters have been
established significantly reduces load-in time
"""
from dataStructures import PCADataset
import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import glob

def saveData(dataset, df, destination):
    try:
        files = glob.glob("/data/time-lags/pca_comps/*.txt")
        for i,file in enumerate(files):
            tmp = file.replace("/data/time-lags/pca_comps/pca_comps_","")
            tmp = int(tmp.replace(".txt",""))
            files[i] = tmp
        files = np.asarray(files)
        start = files.max() + 1
    except:
        start = 0
        
    def save_file(i,item):
        loc = f"/data/time-lags/pca_comps/pca_comps_{i}.txt"
        np.savetxt(loc,item)
        return loc

    cpu_num = os.cpu_count()
    #save data to disk and save location to dataset
    locations = Parallel(n_jobs=cpu_num,verbose=1)(delayed(save_file)(i,item) for i,item in enumerate(dataset,start=start))
    
    locations = np.asarray(locations)
    locations_df = pd.DataFrame(locations,columns=["PCALocation"])
    df = pd.concat([df,locations_df],axis = 1, join = "inner")
    df.to_csv(destination,index=False)

pars_list = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,23]
negatives = [3]
logged = [0,2,3,4,7,8,10,12,13,23]

val_dataset = PCADataset("data/locations/locs_20_spectra_val.csv",
                           pars_list,negatives,logged,scale_bool = False,
                           PCA_loc="scalers/PCA_spec.bin",
                           comp_loc="scalers/comp_spec.bin",
                           spec_scal_loc="scalers/spec_spec.bin")

saveData(np.asarray(val_dataset.data), val_dataset.csv, 
         "data/locations/locs_PCA_comps_val.csv")

train_dataset = PCADataset("data/locations/locs_20_spectra_tra.csv",
                           pars_list,negatives,logged,scale_bool = False,
                           PCA_loc="scalers/PCA_spec.bin",
                           comp_loc="scalers/comp_spec.bin",
                           spec_scal_loc="scalers/spec_spec.bin")

saveData(np.asarray(train_dataset.data), train_dataset.csv, 
         "data/locations/locs_PCA_comps_val.csv")
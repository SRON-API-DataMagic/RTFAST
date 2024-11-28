"""
This program holds custom pytorch data structures for use in this project.
"""
from torch.utils.data import Dataset
from joblib import dump, load, Parallel, delayed
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

class PCADataset(Dataset):
    
    def __init__(self,data_loc,pars_list,negatives,logged,threshold = 1e-11,
                 scale_bool = True, comps = 1,PCA_loc="scalers/PCA_flux.bin",
                 comp_loc="scalers/comp_flux.bin",
                 spec_scal_loc="scalers/spec_flux.bin",force = False,
                 loads = 0,load_size = 1e6, load_pca=False):
        data_table = pd.read_csv(data_loc)
        self.csv = data_table
        self.threshold = threshold
        self.force = force
        self.data_load_size = load_size
        self.PCA_loc = PCA_loc
        self.comp_loc = comp_loc
        self.spec_scal_loc = spec_scal_loc
        self.scale_bool = scale_bool
        self.load_pca = load_pca
        if scale_bool == False:
            self.spec_scaler = load(self.spec_scal_loc)
            self.pca = load(self.PCA_loc)
            self.PCA_scaler = load(self.comp_loc)
        #current number of components loaded
        self.comps = comps
        self.locations = data_table.iloc[:,-1]
        #filter out parameters relevant for emulation
        self.pars = data_table.iloc[:,pars_list].to_numpy()
        self.pars_list = pars_list
        self.negatives = negatives
        self.logged = logged
        self.pars = self.rtdist_to_nn(self.pars)
        self.data = self.data_load(self.locations,loads)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,idx):
        return self.data[idx], self.pars[idx]
    
    def rtdist_to_nn(self,pars):
        """
        Converts rtdist parameters into neural network friendly form.

        Parameters
        ----------
        negatives: list
            list of indexes of parameters to be turned positive due to being
            a negative value in rtdist
        logged: list
            list of indexes of parameters for their logarithm to be inputted
            into the network

        """
        for i, parameter in enumerate(self.pars_list):
            if parameter in self.negatives:
                pars[:,i] = -pars[:,i]
            if parameter in self.logged:
                pars[:,i] = np.log10(pars[:,i])
        return torch.Tensor(pars).float()
    
    def data_load(self,locations,loads=0):
        """
        Loads data from disk and scales it to NN friendly outputs. 
        Automatically splits large loads into 1e6 portions to prevent memory 
        overflow.
        """
        if loads != 0:
            no_loads = loads
        else:
            no_loads = int(np.ceil(len(locations)/self.data_load_size))
        cpu_num = os.cpu_count()
        
        def file_load(file):
            return np.loadtxt(file).reshape(1, -1)
        
        print(f"Loading data in {no_loads} portion(s)")
        for i in range(no_loads):
            print(f"Loading portion {i+1}")
            if (i != (no_loads-1))|(i==0):
                data =(Parallel(n_jobs=cpu_num-1,verbose=1)
                       (delayed(file_load)(file) for file in 
                        locations[int(i*self.data_load_size):(i+1)*int(self.data_load_size)]))
            else:
                if no_loads*self.data_load_size >= len(locations):
                    data = (Parallel(n_jobs=cpu_num-1,verbose=1)
                            (delayed(file_load)(file) for file in 
                             locations[i*int(self.data_load_size):]))
                else:
                    data =(Parallel(n_jobs=cpu_num-1,verbose=1)
                           (delayed(file_load)(file) for file in 
                            locations[int(i*self.data_load_size):(i+1)*int(self.data_load_size)]))
            data = np.concatenate(data,axis=0)
            #make sure all your data is behaving correctly after loading
            if np.any(np.isnan(data))==True:
                print("Data has nan")
            elif np.any(np.isinf(data))==True:
                print("Data has infinities")
            
            if self.load_pca == False:
                data[data<self.threshold] = self.threshold
                data = np.log10(data)
                #make sure all your data is behaving correctly after logging
                if np.any(np.isnan(data))==True:
                    print("Data after log has nan")
                elif np.any(np.isinf(data))==True:
                    print("Data after log has infinities")
                data = self.scale(data)
                #make sure all your data is behaving correctly after scaling
                if np.any(np.isnan(data))==True:
                    print("Data after scale has nan")
                elif np.any(np.isinf(data))==True:
                    print("Data after scale has infinities")
            
            if i == 0:
                overall_data = data
            else:
                overall_data = np.concatenate([overall_data,data],axis=0)
        return torch.Tensor(overall_data).float()
    
    def add_data(self,new_data):
        """
        Loads new data specified in csv new_data. Checks to see if there is 
        overlap between the new data and old data and explicitly excludes
        duplicated data.
        """
        data_table = new_data
        data_table = pd.concat([self.csv,new_data]).drop_duplicates(keep=False)
        locations = data_table.iloc[:,-1].to_numpy()
        
        self.csv = new_data
        self.pars = self.rtdist_to_nn(new_data.iloc[:,self.pars_list].to_numpy())
        self.locations = new_data.iloc[:,-1]
        print(f"Loading {len(locations)} new spectra")
        #load and add new data to dataset
        new_data = self.data_load(locations)
        self.data = torch.concat([self.data,new_data])
        print(f"New length of data is {len(self.data)}")
        return
        
    def scale(self,data):
        data = self.spectra_scaler(data)
        data = self.PCA(data,self.comps)
        data = self.component_scaler(data)
        return data
    
    def spectra_scaler(self,data):
        if self.scale_bool == True:
            self.spec_scaler = StandardScaler()
            data = self.spec_scaler.fit_transform(data)
            dump(self.spec_scaler,self.spec_scal_loc)
        else:
            data = self.spec_scaler.transform(data)
        return data
    
    def PCA(self,data,comp = 1):
        if self.force == True:
            self.pca = PCA(n_components = comp)
            self.pca.fit(data)
            data = self.pca.transform(data)
            print(("Achieved explained variance of"
                   f" {sum(self.pca.explained_variance_ratio_)*100}% with"
                   f" {comp} components"))
            print(("Distribution of explained variance is"
                   f" {self.pca.explained_variance_ratio_}"))
            dump(self.pca,self.PCA_loc)
            return data
        if self.scale_bool == True:
            print(f"Attempting n_comp = {comp}")
            self.pca = PCA(n_components = comp)
            self.pca.fit(data)
            if sum(self.pca.explained_variance_ratio_) < 0.99999:
                print(("Currently achieved explained variance of"
                       f" {sum(self.pca.explained_variance_ratio_)*100}%"))
                self.PCA(data,comp+1)
            else:
                print(f"Successfully describes {.99999*100}% of variance")
                data = self.pca.transform(data)
                dump(self.pca,self.PCA_loc)
        else:
            data = self.pca.transform(data)
        
        return data
    
    def component_scaler(self,data):
        if self.scale_bool == True:
            self.PCA_scaler = StandardScaler()
            data = self.PCA_scaler.fit_transform(data)
            dump(self.PCA_scaler,self.comp_loc)
            self.scale_bool = False
        else:
            data = self.PCA_scaler.transform(data)
        return data
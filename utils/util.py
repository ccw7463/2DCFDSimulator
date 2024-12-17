import os
import numpy as np
import ml_collections
from glob import glob
from configs.config import Config
from tqdm import tqdm
import pandas as pd
import json

class PreprocessData():
    def __init__(self):
        self.path = os.path.join(Config.PREPROCESSED_PATH,"*.csv")
        self.boundary_lst = []
        self.ux_lst = []
        self.uy_lst = []
        self.p_lst = []
        self.boundary_data = None
        self.ux_data = None
        self.uy_data = None
        self.p_data = None
        
    def get_solve_array(self):
        self.filename_lst = sorted(glob(self.path))
        for filepath in tqdm(self.filename_lst):
            df = pd.read_csv(filepath)
            self.boundary_lst.append(df["boundary"])            
            self.ux_lst.append(df["ux"].values)
            self.uy_lst.append(df["uy"].values)
            self.p_lst.append(df["p"].values)
        self.boundary_data = np.array(self.boundary_lst).astype('float32')
        self.ux_data = np.array(self.ux_lst).astype('float32')
        self.uy_data = np.array(self.uy_lst).astype('float32')
        self.p_data = np.array(self.p_lst).astype('float32')

    def get_dataset_config(self):
        self.DataConfig = ml_collections.ConfigDict()
        self.DataConfig.xsize = 101
        self.DataConfig.ysize = 101
        self.DataConfig.ux_max = np.max(self.ux_data)
        self.DataConfig.uy_max = np.max(self.uy_data)
        self.DataConfig.p_max = np.max(self.p_data)

    def scaling_and_resize(self):
        self.get_dataset_config()
        self.boundary_data = self.boundary_data.reshape(-1,self.DataConfig.xsize,self.DataConfig.ysize)
        self.ux_data = self.ux_data.reshape(-1,self.DataConfig.xsize,self.DataConfig.ysize)
        self.ux_data /= self.DataConfig.ux_max
        self.uy_data = self.uy_data.reshape(-1,self.DataConfig.xsize,self.DataConfig.ysize)
        self.uy_data /= self.DataConfig.uy_max
        self.p_data = self.p_data.reshape(-1,self.DataConfig.xsize,self.DataConfig.ysize)
        self.p_data /= self.DataConfig.p_max
        
    def get_operating_dataset(self):
        filenames = [i.split("/")[-1].strip(".csv") for i in self.filename_lst]
        self.char2idx = json.load(open("configs/char2idx.json"))
        self.operating_data = []
        for filename in filenames:
            lst = filename.split("_")
            operating_lst = ["<UX>",str(lst[2]),"<UY>",str(lst[4]),"<DENSITY>",str(1),"<VISCOSITY>",str(1)]
            operating_lst = [self.char2idx[i] for i in operating_lst]
            self.operating_data.append(operating_lst)
        self.operating_data = np.array(self.operating_data)
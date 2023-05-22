import scipy.signal as scipy_signal
import scipy.io as sio
import numpy as np
from scipy.signal import resample
from scipy.signal import filtfilt
from scipy.signal import stft
from scipy import interpolate
from sklearn import preprocessing
import warnings
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

class loadData :
    def __init__(self, path):
	    self.freq = 256
	    self.band = [[4,38]]
	    self.BBAA = []
	    for band_pass in self.band:
	        self.BBAA.append(scipy_signal.butter(3, [i * 2 / self.freq for i in band_pass], btype='bandpass'))
	    self.path = path

    def MakeDataset(self):
        file_list = os.listdir(self.path)

        print("-"*50)
        print("data path check")
        for i in file_list:    # 확인
            print(i, end=' ')

        Data = []
        VAL = []
        ARO = []

        for i in tqdm(file_list, desc="read data"): 
            mat_file = sio.loadmat(self.path+i)
            data = mat_file['data']
            labels = np.array(mat_file['labels'])
            val = labels.T[0]
            val = np.where(val<4, 1, val)
            val = np.where(np.logical_and(val>=4, val<7), 2, val)
            val = np.where(val>=7, 3, val)
            val = val.astype(np.int8)    
            aro = labels.T[ 1]
                
            aro = np.where(aro<4, 1, aro)
            aro = np.where(np.logical_and(aro>=4, aro<7), 2, aro)
            aro = np.where(aro>=7, 3, aro)
            aro = aro.astype(np.int8)
                
            Data.append(data)
            VAL.append(val)
            ARO.append(aro)
                    
        Data = np.concatenate(Data,axis=0)   # 밑으로 쌓아서 하나로 만듬
        VAL = np.concatenate(VAL,axis=0)
        ARO = np.concatenate(ARO,axis=0)
        print(Data.shape, VAL.shape, ARO.shape)

        # eeg preprocessing

        eeg_data = []
        peripheral_data = []

        for i in tqdm(range(len(Data)), desc="preprocess channel"):
            for j in range (40): 
                if(j < 32): # get channels 1 to 32
                    eeg_data.append(Data[i][j])
                else:
                    peripheral_data.append(Data[i][j])

        # set data type, shape
        eeg_data = np.reshape(eeg_data, (len(Data),1,32, 8064))
        
        return eeg_data, VAL, ARO

    def signalProcess(self, _data):
	    data = []
	    for i in range(_data.shape[0]):
	        data.append(self.exponential_running_standardize(_data[i, 0, :, :].T).T)
	    data = np.stack(data, axis=0)
	    return data

    def exponential_running_standardize(self, data, factor_new=0.001, eps=1e-4):
        df = pd.DataFrame(data)
        meaned = df.ewm(alpha=factor_new, axis=0).mean()
        demeaned = df - meaned
        squared = demeaned * demeaned
        square_ewmed = squared.ewm(alpha=factor_new).mean()
        standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
        standardized = np.array(standardized)
        return standardized

    def signalFilter(self, _data):
	    data = []
	    for BBAA in self.BBAA:
	        data.append(filtfilt(BBAA[0], BBAA[1], _data, axis=-1))

	    return np.stack(data, axis=-1)

if __name__ == '__main__':
	pass
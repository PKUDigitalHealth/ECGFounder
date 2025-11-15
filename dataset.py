import numpy as np
import pandas as pd
import wfdb
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.signal import medfilt, iirnotch, filtfilt, butter, resample
from util import filter_bandpass


class LVEF_12lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # leads of MIMIC-ECG are different with leads of HEEDB
        self.fs = 5000 # length of data, 5000 = 500Hz * 10s
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan

    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0 or len(ts) == 0:
            return ts
        t = ts.shape[1] / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
    
        if fs_out == fs_in:
            return ts
        if 2 * fs_out == fs_in:
            return ts[:, ::2]
    
        resampled_ts = np.zeros((ts.shape[0], fs_out))
        x_old = np.linspace(0, t, num=ts.shape[1], endpoint=True) 
        x_new = np.linspace(0, t, num=int(fs_out), endpoint=True) 

        for i in range(ts.shape[0]):
            y_old = ts[i, :]
            f = interp1d(x_old, y_old, kind='linear')
            resampled_ts[i, :] = f(x_new)
    
        return resampled_ts

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = 0
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -1]
        labels = labels.astype(np.float32)
        data = [wfdb.rdsamp(self.ecg_path+hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        result = self.check_nan_in_array(data)
        if result != 0:
            print(hash_file_name)
        data = data.squeeze(0) 
        data = np.transpose(data,  (1, 0))
        data = data[self.lead_indices, :]
        data = filter_bandpass(data, 500) 
        signal = self.resample_unequal(data, sample_rate, self.fs)
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # Convert to torch tensors
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:  
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:  
            labels = labels.unsqueeze(1)
        return signal, labels     
    
class LVEF_12lead_reg_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # leads of MIMIC-ECG are different with leads of HEEDB
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -2]
        labels = torch.tensor([labels], dtype=torch.float32)  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        data = filter_bandpass(data, 500) 
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        return signal, labels     

class LVEF_1lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = 0
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -1]
        labels = labels.astype(np.float32)
        data = [wfdb.rdsamp(self.ecg_path+hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        result = self.check_nan_in_array(data)
        if result != 0:
            print(hash_file_name)
        data = data.squeeze(0) 
        data = np.transpose(data,  (1, 0))
        data = data[self.lead_indices, :]
        data = filter_bandpass(data, 500) 
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # Convert to torch tensors
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:  
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:  
            labels = labels.unsqueeze(1)
        return signal, labels  
    
class LVEF_1lead_reg_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -2]
        labels = torch.tensor([labels], dtype=torch.float32)  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        data = filter_bandpass(data, 500) 
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        return signal, labels     
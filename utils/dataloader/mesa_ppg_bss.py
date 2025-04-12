import os
from glob import glob
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset

from random import shuffle, seed
import pickle

import gc

import numpy as np
from hydra.utils import get_original_cwd

from tqdm import tqdm
from ecgdetectors import Detectors
detectors = Detectors(125)

seed(51465)

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.fs = config.fs
        self.signal_duration = config.signal_duration
        self.get_ecg = config.get_ecg
        self.train_val_split = config.train_val_split

        data_path = config.data_path

        train_data_paths = glob(os.path.join(get_original_cwd(), data_path, '*.dat'))
        train_data = []
        for train_data_path in tqdm(train_data_paths):
            with open(train_data_path, 'rb') as train_outfile:
                new_data = pickle.load(train_outfile, encoding='latin1')
                new_data = np.asarray(new_data)

                _, total_length = np.shape(new_data)

                columns = self.signal_duration * self.fs
                rows = total_length // columns
                
                ecg_data = new_data[0, :]
                # flip ecg if it's upside down
                if (np.nanpercentile(ecg_data, 99) - np.nanmedian(ecg_data)) < (np.nanmedian(ecg_data) - np.nanpercentile(ecg_data, 1)):
                    ecg_data = -ecg_data
                ecg_data = ecg_data[:rows*columns]
                ecg_data = ecg_data.reshape(rows, columns)
                ppg_data = new_data[1, :]
                ppg_data = ppg_data[:rows*columns]
                ppg_data = ppg_data.reshape(rows, columns)

                if self.get_ecg:
                    for i in range(len(ppg_data)):
                        if np.max(ppg_data[i]) != np.min(ppg_data[i]):
                            train_data.append(np.stack([ecg_data[i], ppg_data[i]], axis=-1))
                else:
                    for i in range(len(ppg_data)):
                        if np.max(ppg_data[i]) != np.min(ppg_data[i]):
                            train_data.append(ppg_data[i])

            gc.collect()
                        

        seed(seed)
        shuffle(train_data)

        self.train_split = train_data[:int(len(train_data)*self.train_val_split)]
        print('Train:', len(self.train_split))
        self.val_split = train_data[int(len(train_data)*self.train_val_split):]
        print('Val:', len(self.val_split))

        # load sample plotting data
        sample_data = []
        train_data_paths = glob(os.path.join(get_original_cwd(), data_path, '*.dat'))
        with open(train_data_paths[200], 'rb') as train_outfile:
            new_data = pickle.load(train_outfile, encoding='latin1')
            new_data = np.asarray(new_data)

            _, total_length = np.shape(new_data)

            columns = self.signal_duration * self.fs
            rows = total_length // columns
            
            ecg_data = new_data[0, :]
            if (np.nanpercentile(ecg_data, 99) - np.nanmedian(ecg_data)) < (np.nanmedian(ecg_data) - np.nanpercentile(ecg_data, 1)):
                ecg_data = -ecg_data
            ecg_data = ecg_data[:rows*columns]
            ecg_data = ecg_data.reshape(rows, columns)
            ppg_data = new_data[1, :]
            ppg_data = ppg_data[:rows*columns]
            ppg_data = ppg_data.reshape(rows, columns)
            for i in range(len(ppg_data)):
                sample_data.append(np.stack([ecg_data[i], ppg_data[i]], axis=-1))

        self.x_plot = sample_data[100:102]
        sample = np.asarray(sample_data[100:102])

        # generate ECG R-peak masks for the sample segment (for plotting)
        ecg_sample = sample[:, :, 0]
        masks = []
        for ecg_segment in ecg_sample: 
            hr_peaks = detectors.two_average_detector(ecg_segment)
            hr_peaks = np.asarray(hr_peaks).astype(int)
            mask = np.zeros(len(ecg_segment))
            mask[hr_peaks] = 1
            masks.append(mask)
        mask_sample = np.asarray(masks)
        ppg_sample = sample[:, :, 1]

        samples = []
        for i in range(2):
            samples.append(np.stack([ecg_sample[i], mask_sample[i], ppg_sample[i]], axis=-1))
        
        self.x_plot = samples
    
    
    def setup(self, stage):
        self.train_data = BiosigDataset(self.train_split, self.get_ecg)
        self.val_data = BiosigDataset(self.val_split, self.get_ecg)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, 
                            num_workers=self.num_workers, shuffle=False)


class BiosigDataset(Dataset):
    def __init__(self, data, get_ecg):
        self.data = data
        self.get_ecg = get_ecg

    def __len__(self):
        return len(self.data)
    
    def _local_normalize(self, x):
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        
        return x

    def __getitem__(self, idx):
        
        if self.get_ecg:
            ecg, ppg = self.data[idx][:, 0], self.data[idx][:, 1]
            ecg, ppg = torch.tensor(ecg), torch.tensor(ppg)
            ecg = self._local_normalize(ecg)
            ppg = self._local_normalize(ppg)
            return ecg.float().unsqueeze_(0), ppg.float().unsqueeze_(0)
        
        else:
            ppg = self.data[idx]
            ppg = torch.tensor(ppg)
            ppg = self._local_normalize(ppg)
            return ppg.float().unsqueeze_(0)
    
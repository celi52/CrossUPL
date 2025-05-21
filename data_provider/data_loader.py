import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Dataset_PJM_OT(Dataset):
    def __init__(self, root_path, flag='train', pjm_sub_name='AEP', 
                 pjm_feature='elevation', size=None,
                 features='S', data_path='PJM_data.csv',
                 target='OT', scale=False, timeenc=0, freq='h', 
                 seasonal_patterns=None,
                 weather_train=False,
                 sub_train=False,
                 ext_feature='tmpf',
                 model_id='PJM'):
        
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init 
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.pjm_sub_name = pjm_sub_name
        self.pjm_feature = pjm_feature
        self.weather_train = weather_train
        self.sub_train = sub_train
        self.ext_feature = ext_feature
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col=['date'])
        # df_raw = df_raw.xs(self.pjm_sub_name, level='sub')
        if not self.sub_train:
            df_raw = df_raw[[self.pjm_sub_name]]
        
        if self.weather_train:
            df_weather = pd.read_csv(os.path.join(self.root_path, 'PJM_weather.csv'), index_col=['date'])
            df_weather = df_weather[df_weather['sub']==self.pjm_sub_name]
            if self.ext_feature == 'all':
                df_weather = df_weather[['tmpf', 'relh', 'sknt', 'alti', 'vsby']]
            else:
                df_weather = df_weather[[self.ext_feature]]
            df_raw = pd.concat([df_raw, df_weather], axis=1)

        df_before_2022 = df_raw[(df_raw.index >= '2019-01-01') & (df_raw.index <= '2019-03-31')]
        df_before_2023 = df_raw[df_raw.index < '2019-05-01']
        df_before_2024 = df_raw[df_raw.index < '2024-06-01']
        # Get the first and last index of the filtered DataFrame
        first_2022_index = df_before_2022.index.get_loc(df_before_2022.index[0])
        last_2022_index = df_before_2022.index.get_loc(df_before_2022.index[-1]) + 1
        last_2023_index = df_before_2023.index.get_loc(df_before_2023.index[-1]) + 1
        last_2024_index = df_before_2024.index.get_loc(df_before_2024.index[-1]) + 1
        
        border1s = [0, last_2022_index - self.seq_len, last_2023_index - self.seq_len]
        border2s = [last_2022_index, last_2023_index, last_2024_index]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.pjm_sub_name]]
            
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        df_raw.reset_index(inplace=True)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        shape_row, shape_coulumn = self.data_x.shape
        self.data_x = self.data_x.reshape(shape_row//24, 24, shape_coulumn)
        self.data_y = self.data_y.reshape(shape_row//24, 24, shape_coulumn)
        data_stamp_column = self.data_stamp.shape[1]
        self.data_stamp = self.data_stamp.reshape(shape_row//24, 24, data_stamp_column)
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len//24
        r_begin = s_end - self.label_len//24
        r_end = r_begin + self.label_len//24 + self.pred_len//24
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len//24 - self.pred_len//24 + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_OT(Dataset):
    # dataloader for NYISO MISO ISNE
    def __init__(self, root_path, flag='train', pjm_sub_name='ZONA ', 
                 pjm_feature='elevation', size=None,
                 features='S', data_path='NYIS_OT_data.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None,
                 weather_train=False,
                 sub_train=False,
                 ext_feature='tmpf',
                 model_id='NYIS'):
        
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init 
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.pjm_sub_name = pjm_sub_name
        self.pjm_feature = pjm_feature
        self.weather_train = weather_train
        self.sub_train = sub_train
        self.ext_feature = ext_feature
        self.model_id = model_id
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col=['date'])
        # df_raw = df_raw.xs(self.pjm_sub_name, level='sub')
        if not self.sub_train:
            df_raw = df_raw[[self.pjm_sub_name]]
        
        if self.weather_train:
            df_weather = pd.read_csv(os.path.join(self.root_path, self.model_id + '_weather_data.csv'), index_col=['date'])
            if self.pjm_sub_name[0] == 'Z':
                df_weather = df_weather[df_weather['sub']==self.pjm_sub_name]
            else:
                df_weather = df_weather[df_weather['sub']==int(self.pjm_sub_name)]
            if self.ext_feature == 'all':
                df_weather = df_weather[['tmpf', 'relh', 'sknt', 'alti', 'vsby']]
            else:
                df_weather = df_weather[[self.ext_feature]]
            df_raw = pd.concat([df_raw, df_weather], axis=1)

        # interpolate missing values
        # df_raw.interpolate(method='linear', inplace=True)
        # Filter to select rows before the year 2022
        df_before_2022 = df_raw[(df_raw.index >= '2019-01-01') & (df_raw.index < '2023-01-01')]
        df_before_2023 = df_raw[df_raw.index < '2024-01-01']
        df_before_2024 = df_raw[df_raw.index < '2024-12-31']
        # Get the first and last index of the filtered DataFrame
        first_2022_index = df_before_2022.index.get_loc(df_before_2022.index[0])
        last_2022_index = df_before_2022.index.get_loc(df_before_2022.index[-1]) + 1
        last_2023_index = df_before_2023.index.get_loc(df_before_2023.index[-1]) + 1
        last_2024_index = df_before_2024.index.get_loc(df_before_2024.index[-1]) + 1
        
        # border1s = [0, (365+366+365) * 24 - self.seq_len, (365+366+365+365) * 24 - self.seq_len]
        border1s = [0, last_2022_index - self.seq_len, last_2023_index - self.seq_len]
        border2s = [last_2022_index, last_2023_index, last_2024_index]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns
            # cols_data = df_raw.columns
            # cols_data = [df_raw.columns[3], df_raw.columns[-1]]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.pjm_sub_name]]
            
        data = df_data.values
        df_raw.reset_index(inplace=True)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        shape_row, shape_coulumn = self.data_x.shape
        self.data_x = self.data_x.reshape(shape_row//24, 24, shape_coulumn)
        self.data_y = self.data_y.reshape(shape_row//24, 24, shape_coulumn)
        data_stamp_column = self.data_stamp.shape[1]
        self.data_stamp = self.data_stamp.reshape(shape_row//24, 24, data_stamp_column)
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len//24
        r_begin = s_end - self.label_len//24
        r_end = r_begin + self.label_len//24 + self.pred_len//24
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len//24 - self.pred_len//24 + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

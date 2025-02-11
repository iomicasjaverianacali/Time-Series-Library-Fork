import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
from utils.augmentation import run_augmentation_single
from datetime import timedelta

import os
from tsa_methane import DataManager

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
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
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

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

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
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
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
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
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

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

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class MethaneSynthGapsSingleVisit(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
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
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        interpolate = True
        smooth = True

        mydm = DataManager()
        filled_nans_all_visits_meas = mydm.make_meas_with_nans(path_to_root_folder_all_meas='/'.join(self.root_path.split("/")[:-1]))

        n_visit = 0 # 2ndVisit

        df_raw = filled_nans_all_visits_meas[n_visit]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        nan_indexes = df_raw[df_raw[self.target].isna()].index

        consecutive_groups = self.group_consecutive(list(nan_indexes))
        y_signals = self.fill_nans_w_signals(consecutive_groups)

        for cnt, group in enumerate(consecutive_groups):
            df_raw.loc[group, self.target] = y_signals[cnt]
            
            start_date = df_raw.loc[group[0] - 1, 'date'] if group[0] > 0 else pd.Timestamp('2023-01-01 00:00:00')
            for i, idx in enumerate(group):
                df_raw.loc[idx, 'date'] = start_date + timedelta(seconds=0.5 * (i + 1))
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

            if smooth:
                df_data = df_data.rolling(window=10, min_periods=1).mean()

            if interpolate:
                M = 10000
                N = df_data[self.target].values.shape[0]

                time_delta = timedelta(seconds=0.5)

                # FFT of the signal
                fft_y = np.fft.fft(df_data['METHANE'].values)

                # Zero-padding in the frequency domain
                zero_pad = M - N
                fft_y_padded = np.concatenate([
                                fft_y[:N // 2].squeeze(),  # First half of FFT
                                np.zeros(zero_pad),  # Zeros in the middle
                                fft_y[N // 2:].squeeze()  # Second half of FFT
                            ])

                # Inverse FFT to get interpolated signal
                y_interpolated = np.fft.ifft(fft_y_padded).real  # Take the real part

                # Scale the interpolated signal to match the original amplitude
                scale_factor = M / N
                y_interpolated *= scale_factor

                # Create a new pandas DataFrame for the interpolated signal
                df_data = pd.DataFrame({f'{self.target}': y_interpolated})

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if interpolate:
            # Generate new datetime entries
            x_new = [df_stamp['date'][0] + i * time_delta / (M / N) for i in range(M)]
            df_stamp = pd.DataFrame({f'date': x_new})

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
    
    def fill_nans_w_signals(self, consecutive_groups):
        peaks_per_sec = 5/(60*5)
        y_signals = []
        for g in consecutive_groups:
            time_interval = len(g)/0.5
            N_peaks = int(time_interval*peaks_per_sec)

            x = np.linspace(0, len(g), len(g))

            y_combined = np.zeros((len(g),))

            for i in range(N_peaks):
                #idx_synth_peak = np.random.randint(low=10, high=len(consecutive_groups[0])-30)
                idx_synth_peak = int(i*len(g)/N_peaks)
                factor=np.random.rand()

                lda = 0.5
                A=1000*factor
                y = A*np.exp(-x/lda)
                
                y_combined = y_combined + np.roll(y, idx_synth_peak)
            
            y_signals.append(y_combined)

        return y_signals
    
    def group_consecutive(self, indices):
        grouped = []
        temp_group = [indices[0]]  # Start with the first index

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                temp_group.append(indices[i])  # Continue the current group
            else:
                grouped.append(temp_group)  # Finalize the current group
                temp_group = [indices[i]]  # Start a new group

        grouped.append(temp_group)  # Add the last group
        return grouped

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class MethaneSingleVisit(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
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
        self.__read_data__()
    
    def group_consecutive(self, indices):
        grouped = []
        temp_group = [indices[0]]  # Start with the first index

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                temp_group.append(indices[i])  # Continue the current group
            else:
                grouped.append(temp_group)  # Finalize the current group
                temp_group = [indices[i]]  # Start a new group

        grouped.append(temp_group)  # Add the last group
        return grouped

    def __read_data__(self):
        self.scaler = StandardScaler()

        list_of_discontinous_ts = []
        lens_of_discontinous_ts = []
        list_of_discontinuous_timefeatures = []
        lens_of_discontinous_timefeatures = []

        interpolate = True
        smooth = True

        n_visit = 2

        if n_visit == 2:
            folder = f"{self.root_path}/2ndVisit"
        if n_visit == 3:
            folder = f"{self.root_path}/3rdVisit"
        if n_visit == 4:
            folder = f"{self.root_path}/4thVisit"

        files = os.listdir(folder)
        files.sort()
        for file_i in files:
            try:
                df_raw = pd.read_csv(os.path.join(os.path.join(self.root_path, folder), file_i), header=None)
                df_raw.columns = ['date', 'METHANE']

                if self.features == 'M' or self.features == 'MS':
                    cols_data = df_raw.columns[1:]
                    df_data = df_raw[cols_data]
                elif self.features == 'S':
                    df_data = df_raw[[self.target]]

                    if smooth:
                        df_data = df_data.rolling(window=10, min_periods=1).mean()

                    if interpolate:
                        M = 1000
                        N = df_data[self.target].values.shape[0]

                        time_delta = timedelta(seconds=0.5)

                        # FFT of the signal
                        fft_y = np.fft.fft(df_data['METHANE'].values)

                        # Zero-padding in the frequency domain
                        zero_pad = M - N
                        fft_y_padded = np.concatenate([
                                fft_y[:N // 2].squeeze(),  # First half of FFT
                                np.zeros(zero_pad),  # Zeros in the middle
                                fft_y[N // 2:].squeeze()  # Second half of FFT
                            ])

                        # Inverse FFT to get interpolated signal
                        y_interpolated = np.fft.ifft(fft_y_padded).real  # Take the real part

                        # Scale the interpolated signal to match the original amplitude
                        scale_factor = M / N
                        y_interpolated *= scale_factor

                        # Create a new pandas DataFrame for the interpolated signal
                        df_data = pd.DataFrame({f'{self.target}': y_interpolated})

                if self.scale:
                    self.scaler.fit(df_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values

                df_stamp = df_raw[['date']]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)

                if interpolate:
                    # Generate new datetime entries
                    x_new = [df_stamp['date'][0] + i * time_delta / (M / N) for i in range(M)]
                    df_stamp = pd.DataFrame({f'date': x_new})

                if self.timeenc == 0:
                    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    data_stamp = df_stamp.drop(['date'], 1).values
                elif self.timeenc == 1:
                    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)
                    
                list_of_discontinuous_timefeatures.append(data_stamp)
                lens_of_discontinous_timefeatures.append(data_stamp.shape[0])
                list_of_discontinous_ts.append(data)
                lens_of_discontinous_ts.append(data.shape[0])
            except FileNotFoundError or IsADirectoryError:
                continue

        self.list_of_discontinous_ts = list_of_discontinous_ts
        self.lens_of_discontinous_ts = lens_of_discontinous_ts
        self.list_of_discontinuous_timefeatures = list_of_discontinuous_timefeatures
        self.lens_of_discontinous_timefeatures = lens_of_discontinous_timefeatures

        num_train = int(len(self.list_of_discontinous_ts) * 0.7)
        num_test = int(len(self.list_of_discontinous_ts) * 0.2)
        num_vali = len(self.list_of_discontinous_ts) - num_train - num_test

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(self.lens_of_discontinous_timefeatures)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.list_of_discontinous_ts = self.list_of_discontinous_ts[border1:border2]
        self.list_of_discontinuous_timefeatures = self.list_of_discontinuous_timefeatures[border1:border2]
        self.lens_of_discontinous_ts = self.lens_of_discontinous_ts[border1:border2]
        self.lens_of_discontinous_timefeatures = self.lens_of_discontinous_timefeatures[border1:border2]

        print(f"Lengths: {self.lens_of_discontinous_ts}")
        #if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)        
    
    def find_discontinous_ts_number_and_timestep_index(self, I):
        accumulated_length = 0
        
        for disc_ts_num, length in enumerate(self.lens_of_discontinous_ts):
            if accumulated_length <= I < accumulated_length + length:
                # Calculate the index within the disc_ts_num
                measurement_index = I - accumulated_length
                return disc_ts_num, measurement_index
            accumulated_length += length
        
        return None, None  # If I is out of bounds

    def __getitem__(self, index):
        n_discontinous_ts, idx_in_meas = self.find_discontinous_ts_number_and_timestep_index(index)

        try:
            if idx_in_meas > self.lens_of_discontinous_ts[n_discontinous_ts] - self.seq_len - self.pred_len:
                n_discontinous_ts = n_discontinous_ts + 1
                idx_in_meas = 0
        except Exception as e:
            print(e)

        s_begin = idx_in_meas
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.list_of_discontinous_ts[n_discontinous_ts][s_begin:s_end]
        seq_y = self.list_of_discontinous_ts[n_discontinous_ts][r_begin:r_end]
        
        seq_x_mark = self.list_of_discontinuous_timefeatures[n_discontinous_ts][s_begin:s_end]
        seq_y_mark = self.list_of_discontinuous_timefeatures[n_discontinous_ts][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        total_len = 0
        for this_len in self.lens_of_discontinous_ts:
            total_len = total_len + this_len - self.seq_len - self.pred_len + 1
        return total_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class MethaneAllVisits(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
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
        self.__read_data__()
    
    def group_consecutive(self, indices):
        grouped = []
        temp_group = [indices[0]]  # Start with the first index

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                temp_group.append(indices[i])  # Continue the current group
            else:
                grouped.append(temp_group)  # Finalize the current group
                temp_group = [indices[i]]  # Start a new group

        grouped.append(temp_group)  # Add the last group
        return grouped

    def __read_data__(self):
        """
        Datasets are organized in a specific folder structure:

        -root_path
            -1stVisit
                -meas_merged_1.csv
                .
                .
                .
                -meas_merged_n1.csv
            -2ndVisit
                --meas_merged_1.csv
                .
                .
                .
                -meas_merged_n2.csv
            .
            .
            .
            -[Lth]Visit
                --meas_merged_1.csv
                .
                .
                .
                -meas_merged_nL.csv
        """
        self.scaler = StandardScaler()

        subfolders = [f.path for f in os.scandir(self.root_path) if f.is_dir()]
        subfolders.sort()

        list_of_discontinous_ts = []
        lens_of_discontinous_ts = []
        list_of_discontinuous_timefeatures = []
        lens_of_discontinous_timefeatures = []

        interpolate = False
        smooth = True

        for folder in subfolders:
            if 'Visit' not in folder:
                continue
            
            files = os.listdir(folder)
            files.sort()
            for file_i in files:
                try:
                    df_raw = pd.read_csv(os.path.join(os.path.join(self.root_path, folder), file_i), header=None)
                    df_raw.columns = ['date', 'METHANE']

                    if self.features == 'M' or self.features == 'MS':
                        cols_data = df_raw.columns[1:]
                        df_data = df_raw[cols_data]
                    elif self.features == 'S':
                        df_data = df_raw[[self.target]]

                        if smooth:
                            df_data = df_data.rolling(window=10, min_periods=1).mean()

                        if interpolate:
                            M = 1000
                            N = df_data[self.target].values.shape[0]

                            time_delta = timedelta(seconds=0.5)

                            # FFT of the signal
                            fft_y = np.fft.fft(df_data['METHANE'].values)

                            # Zero-padding in the frequency domain
                            zero_pad = M - N
                            fft_y_padded = np.concatenate([
                                fft_y[:N // 2].squeeze(),  # First half of FFT
                                np.zeros(zero_pad),  # Zeros in the middle
                                fft_y[N // 2:].squeeze()  # Second half of FFT
                            ])

                            # Inverse FFT to get interpolated signal
                            y_interpolated = np.fft.ifft(fft_y_padded).real  # Take the real part

                            # Scale the interpolated signal to match the original amplitude
                            scale_factor = M / N
                            y_interpolated *= scale_factor

                            # Create a new pandas DataFrame for the interpolated signal
                            df_data = pd.DataFrame({f'{self.target}': y_interpolated})

                    if self.scale:
                        self.scaler.fit(df_data.values)
                        data = self.scaler.transform(df_data.values)
                    else:
                        data = df_data.values

                    df_stamp = df_raw[['date']]
                    df_stamp['date'] = pd.to_datetime(df_stamp.date)

                    if interpolate:
                        # Generate new datetime entries
                        x_new = [df_stamp['date'][0] + i * time_delta / (M / N) for i in range(M)]
                        df_stamp = pd.DataFrame({f'date': x_new})

                    if self.timeenc == 0:
                        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                        data_stamp = df_stamp.drop(['date'], 1).values
                    elif self.timeenc == 1:
                        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                        data_stamp = data_stamp.transpose(1, 0)
                    
                    list_of_discontinuous_timefeatures.append(data_stamp)
                    lens_of_discontinous_timefeatures.append(data_stamp.shape[0])
                    list_of_discontinous_ts.append(data)
                    lens_of_discontinous_ts.append(data.shape[0])
                except FileNotFoundError or IsADirectoryError:
                    continue

        self.list_of_discontinous_ts = list_of_discontinous_ts
        self.lens_of_discontinous_ts = lens_of_discontinous_ts
        self.list_of_discontinuous_timefeatures = list_of_discontinuous_timefeatures
        self.lens_of_discontinous_timefeatures = lens_of_discontinous_timefeatures

        num_train = int(len(self.list_of_discontinous_ts) * 0.7)
        num_test = int(len(self.list_of_discontinous_ts) * 0.2)
        num_vali = len(self.list_of_discontinous_ts) - num_train - num_test

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(self.lens_of_discontinous_timefeatures)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.list_of_discontinous_ts = self.list_of_discontinous_ts[border1:border2]
        self.list_of_discontinuous_timefeatures = self.list_of_discontinuous_timefeatures[border1:border2]
        self.lens_of_discontinous_ts = self.lens_of_discontinous_ts[border1:border2]
        self.lens_of_discontinous_timefeatures = self.lens_of_discontinous_timefeatures[border1:border2]

        print(f"Lengths: {self.lens_of_discontinous_ts}")
        #if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)        
    
    def find_discontinous_ts_number_and_timestep_index(self, I):
        accumulated_length = 0
        
        for disc_ts_num, length in enumerate(self.lens_of_discontinous_ts):
            if accumulated_length <= I < accumulated_length + length:
                # Calculate the index within the disc_ts_num
                measurement_index = I - accumulated_length
                return disc_ts_num, measurement_index
            accumulated_length += length
        
        return None, None  # If I is out of bounds

    def __getitem__(self, index):
        n_discontinous_ts, idx_in_meas = self.find_discontinous_ts_number_and_timestep_index(index)

        try:
            if idx_in_meas > self.lens_of_discontinous_ts[n_discontinous_ts] - self.seq_len - self.pred_len:
                n_discontinous_ts = n_discontinous_ts + 1
                idx_in_meas = 0
        except Exception as e:
            print(e)

        s_begin = idx_in_meas
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.list_of_discontinous_ts[n_discontinous_ts][s_begin:s_end]
        seq_y = self.list_of_discontinous_ts[n_discontinous_ts][r_begin:r_end]
        
        seq_x_mark = self.list_of_discontinuous_timefeatures[n_discontinous_ts][s_begin:s_end]
        seq_y_mark = self.list_of_discontinuous_timefeatures[n_discontinous_ts][r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        total_len = 0
        for this_len in self.lens_of_discontinous_ts:
            total_len = total_len + this_len - self.seq_len - self.pred_len + 1
        return total_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


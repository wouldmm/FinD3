import os
from datetime import datetime

import numpy as np
import pandas as pd
import re
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
import warnings

warnings.filterwarnings('ignore')

class CS300_Dataset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='stock.csv',
                 target='close', scale=True, freq='d'):
        self.df = pd.read_csv(os.path.join(root_path,
                                 data_path), index_col=0)
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.target = target
        self.scale = True
        self.scaler = StandardScaler()
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # 把target放到最后一列
        target_col = self.df.pop(self.target)
        self.df.insert(loc=self.df.shape[1], column=self.target, value=target_col, allow_duplicates=False)

        # 处理时间和code列
        all_date = pd.DataFrame(self.df['time'].unique(), columns=['time'])
        self.df = self.df.set_index('time')
        self.feature_names = self.df.columns.drop('code').to_list()
        self.unique_stocks = self.df['code'].unique()

        self.num_stocks = len(self.unique_stocks)  # 股票数
        self.num_features = len(self.feature_names)  # 特征数
        self.feature = self.df[self.feature_names]  # 特征数据
        self.length = len(self.df.index.unique())  # 总时间长度

        # 划分数据集
        num_train = int(self.length * 0.6)
        num_test = int(self.length * 0.2)
        num_vali = int(self.length * 0.2)

        border1s = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.all_data = np.zeros((self.num_stocks, self.length, self.num_features))
        self.all_label = np.zeros((self.num_stocks, self.length))
        grouped = self.df.groupby('code')
        for i, (code, group) in enumerate(grouped):
            group[self.target] = group[self.target].pct_change().fillna(0)
            data = group[self.feature_names]
            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(data.values)
            self.all_data[i, :, :] = data
            self.all_label[i, :] = group[self.target].values

            if i == 0:
                print('#single EOD data shape:', data.shape)
                masks = np.ones([self.num_stocks, data.shape[0]], dtype=np.float32)
                ground_truth = np.zeros([self.num_stocks, data.shape[0]], dtype=np.float32)
                base_price = np.zeros([self.num_stocks,data.shape[0]], dtype=np.float32)

            for row in range(data.shape[0]):
                if abs(data[row][-1]) < 1e-8:
                    masks[i][row] = 0.0
                elif row > 0 and abs(data[row - 1][-1]) > 1e-8:
                    ground_truth[i][row] = (data[row][-1] - data[row - 1][-1]) / data[row - 1][-1]
                for col in range(data.shape[1]):
                    if abs(data[row][col]) < 1e-8:
                        data[row][col] = 1.0  # 空值处理

            base_price[i, :] = data[:, -1]

        self.data_stamp = all_date[border1:border2]
        self.data_stamp.loc[:, 'time'] = pd.to_datetime(self.data_stamp['time'])
        self.data_stamp = time_features(pd.to_datetime(self.data_stamp['time'].values), freq=freq)
        self.data_stamp = self.data_stamp.transpose(1, 0)

        self.data_x = self.all_data[:, border1:border2]
        self.data_y = self.all_data[:, border1:border2]
        self.label = self.all_label[:, border1:border2]
        # plot_histogram(self.data_x, str(self.set_type)+"cs300")

        self.masks = masks
        self.ground_truth = ground_truth
        self.base_price = base_price

    def __len__(self):
        return self.data_x.shape[1] - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[:, s_begin:s_end, :]
        # seq_y = self.label[:, r_begin:r_end]
        seq_y = self.data_y[:, r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        mask_batch = self.masks[:, idx: idx + self.seq_len + self.pred_len]
        mask_batch = np.min(mask_batch, axis=1)
        mask_batch = np.expand_dims(mask_batch, axis=1)

        price_batch = self.base_price[:, r_begin - 1: r_end - 1]
        gt_batch = self.ground_truth[:, r_begin: r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask_batch, price_batch, gt_batch

class CS300_Dataset_baseline(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='stock.csv',
                 target='close', scale=True, code=None, freq='d'):
        self.df = pd.read_csv(os.path.join(root_path,
                                 data_path), index_col=0)
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.target = target
        self.scale = scale
        self.scaler = StandardScaler()
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.df = self.df[self.df['code'] == code]
        self.df.drop(['code'], axis=1, inplace=True)
        # 把target放到最后一列
        target_col = self.df.pop(self.target)
        self.df.insert(loc=self.df.shape[1], column=self.target, value=target_col, allow_duplicates=False)
        # 计算return
        self.df[self.target] = self.df[self.target].pct_change().fillna(0)

        # 处理时间和code列
        all_date = self.df[['time']]
        self.df = self.df.set_index('time')
        self.length = len(self.df.index.unique())  # 总时间长度

        # 划分数据集
        num_train = int(self.length * 0.6)
        num_test = int(self.length * 0.2)
        num_vali = int(self.length * 0.2)

        border1s = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.all_labels = np.zeros(self.length)

        if self.scale:
            train_data = self.df[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.df = self.scaler.transform(self.df.values)

        self.data_stamp = all_date[border1:border2]
        self.data_stamp['time'] = pd.to_datetime(self.data_stamp['time'])
        self.data_stamp = time_features(pd.to_datetime(self.data_stamp['time'].values), freq=freq)
        self.data_stamp = self.data_stamp.transpose(1, 0)

        self.data_x = self.df[border1:border2]
        self.data_y = self.df[border1:border2]
        self.label = self.all_labels[border1:border2]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        # seq_y = self.label[r_begin:r_end]
        seq_y = self.data_x[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

class Dataset_baseline_why(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=1, freq='B'):
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
        self.__read_data__()

    def __read_data__(self):
        # self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
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

        # if self.scale:
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     # print(self.scaler.mean_)
        #     # exit()
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values

        data = df_data.values

        closing_prices = data[:, -1]
        gt_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.gt_return = gt_returns[border1:border2 - 1]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]  # 当日收盘价
        seq_return = self.data_y[r_begin - 1:r_end - 1]  # 前一日的收盘价
        # gt_returns = self.gt_return[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_return

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class NASDAQ_Dataset(Dataset):
    def __init__(self, root_path, market_name,
                 flag='train', size=None,
                 features='MS', data_path='stock_sup/2013-01-01',
                 target='Close', scale=True, timeenc=0, freq='d'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 16
            self.label_len = 8
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # self.steps = self.pred_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()
        self.timeenc = timeenc
        self.freq = freq

        self.market_name = market_name
        self.tickers_fname = market_name+'_tickers_qualify_dr-0.98_min-5_smooth.csv'
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # read tickers' name
        tickers = np.genfromtxt(
            os.path.join(self.root_path, self.data_path, '..', self.tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print('#tickers selected:', len(tickers))

        eod_data = []
        masks = []
        ground_truth = []
        base_price = []
        data_time_stamp = []

        if self.market_name == 'TSE':
            trade_dates = 1188
            valid_index = 693
            test_index = 924
        else:
            trade_dates = 1245
            valid_index = 756
            test_index = 1008
        border1s = [0, valid_index - self.seq_len, test_index - self.seq_len]
        border2s = [valid_index, test_index, trade_dates]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # read tickers' eod data
        length = len(tickers)
        # length = 200
        for index, ticker in enumerate(tickers):
            if index >= length:
                break
            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path,
                                              self.market_name + '_' + ticker + '_1.csv'))

            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
            # print(cols)
            if self.market_name == 'NASDAQ':
                # remove the last day since lots of missing data
                df_raw = df_raw[:-1]
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
            data = df_data.values
            # if self.scale:
            #     train_data = data[border1s[0]:border2s[0]]
            #     self.scaler.fit(train_data)
            #     data = self.scaler.transform(data)

            data = data[border1:border2]

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(columns=['date']).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            if index == 0:
                print('#single EOD data shape:', data.shape)
                # [股票数*交易日数*5[5-day,10-day,20-day,30-day,Close]]
                eod_data = np.zeros([length, data.shape[0], data.shape[1]], dtype=np.float32)
                masks = np.ones([length, data.shape[0]], dtype=np.float32)
                ground_truth = np.zeros([length, data.shape[0]], dtype=np.float32)
                base_price = np.zeros([length, data.shape[0]], dtype=np.float32)

            for row in range(data.shape[0]):
                if abs(data[row][-1] + 1234) < 1e-8:
                    masks[index][row] = 0.0
                elif row > 0 and abs(data[row - 1][-1] + 1234) > 1e-8:
                    ground_truth[index][row] = (data[row][-1] - data[row - 1][-1]) / data[row - 1][-1]
                for col in range(data.shape[1]):
                    if abs(data[row][col] + 1234) < 1e-8:
                        data[row][col] = 1.0  # 空值处理
            eod_data[index, :, :] = data
            base_price[index, :] = data[:, -1]
            data_time_stamp.append(data_stamp)
        data_stamp = np.array(data_time_stamp)
        print('#eod_data shape:', eod_data.shape)
        print('#masks shape:', masks.shape)
        print('#ground_truth shape:', ground_truth.shape)
        print('#base_price shape:', base_price.shape)
        print('#data_stamp shape:', data_stamp.shape)
        self.data_x = eod_data

        self.masks = masks
        self.ground_truth = ground_truth
        self.base_price = base_price
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.seq_len > 16:
            mask_seq_len = 16
        else:
            mask_seq_len = self.seq_len

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[:, s_begin:s_end, :]
        seq_y = self.data_x[:, r_begin:r_end, :]
        seq_x_mark = self.data_stamp[:, s_begin:s_end, :]
        seq_y_mark = self.data_stamp[:, r_begin:r_end, :]

        mask_batch = self.masks[:, index: index + self.seq_len + self.pred_len]
        mask_batch = np.min(mask_batch, axis=1)
        mask_batch = np.expand_dims(mask_batch, axis=1)

        price_batch = self.base_price[:, r_begin - 1: r_end - 1]
        gt_batch = self.ground_truth[:, r_begin: r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask_batch, price_batch, gt_batch
        # return seq_x, seq_y, seq_x_mark, seq_y_mark
    def __len__(self):
        return self.data_x.shape[1] - self.seq_len - self.pred_len + 1

class NASDAQ_Dataset_baseline(Dataset):
    def __init__(self, root_path, market_name, ticket,
                 flag='train', size=None,
                 features='MS', data_path='stock_sup/2013-01-01',
                 target='Close', scale=True, timeenc=0, freq='d'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 16
            self.label_len = 8
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # self.steps = self.pred_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()
        self.timeenc = timeenc
        self.freq = freq

        self.market_name = market_name
        self.tickers_fname = ticket
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # read tickers' name
        if self.market_name == 'TSE':
            trade_dates = 1188
            valid_index = 693
            test_index = 924
        else:
            trade_dates = 1245
            valid_index = 756
            test_index = 1008
        border1s = [0, valid_index - self.seq_len, test_index - self.seq_len]
        border2s = [valid_index, test_index, trade_dates]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # read tickers' eod data
        # length = len(tickers)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path,
                                          self.market_name + '_' + self.tickers_fname + '_1.csv'))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        if self.market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            df_raw = df_raw[:-1]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        data = df_data.values
        # if self.scale:
        #     train_data = data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data)
        #     data = self.scaler.transform(data)

        data = data[border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        print('#single EOD data shape:', data.shape)
        # [股票数*交易日数*5[5-day,10-day,20-day,30-day,Close]]
        masks = np.ones([data.shape[0]], dtype=np.float32)
        ground_truth = np.zeros([data.shape[0]], dtype=np.float32)

        for row in range(data.shape[0]):
            if abs(data[row][-1] + 1234) < 1e-8:
                masks[row] = 0.0
            elif row > 0 and abs(data[row - 1][-1] + 1234) > 1e-8:
                ground_truth[row] = (data[row][-1] - data[row - 1][-1]) / data[row - 1][-1]
            for col in range(data.shape[1]):
                if abs(data[row][col] + 1234) < 1e-8:
                    data[row][col] = 1.0  # 空值处理
        eod_data = data
        base_price = data[:, -1]

        print('#eod_data shape:', eod_data.shape)
        print('#masks shape:', masks.shape)
        print('#ground_truth shape:', ground_truth.shape)
        print('#base_price shape:', base_price.shape)
        print('#data_stamp shape:', data_stamp.shape)
        self.data_x = eod_data

        self.masks = masks
        self.ground_truth = ground_truth
        self.base_price = base_price
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.seq_len > 16:
            mask_seq_len = 16
        else:
            mask_seq_len = self.seq_len

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_x[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end, :]
        seq_y_mark = self.data_stamp[r_begin:r_end, :]

        mask_batch = self.masks[index: index + self.seq_len + self.pred_len]
        mask_batch = np.min(mask_batch)
        mask_batch = np.expand_dims(mask_batch, axis=0)

        price_batch = self.base_price[r_begin - 1: r_end - 1]
        gt_batch = self.ground_truth[r_begin: r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask_batch, price_batch, gt_batch

    def __len__(self):
        return self.data_x.shape[0] - self.seq_len - self.pred_len + 1


def plot_histogram(data, title):
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[2]):  # 假设data.shape[2]是特征维度
        plt.subplot(1, data.shape[2], i + 1)
        plt.hist(data[:, :, i].flatten(), bins=50, alpha=0.5, label=f'Feature {i}')
        plt.title(f'{title} - Feature {i}')
    plt.legend()
    plt.savefig(f'{title}_histogram.png')
    plt.close()


import os

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

from timefeatures import time_features
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from tqdm import tqdm

def get_data(root_path, market_name,
             flag='train', size=None,
             features='MS', data_path='stock_sup/2013-01-01',
             target='Close', scale=True, timeenc=0, freq='d'):
    if size == None:
        seq_len = 16
        label_len = 8
        pred_len = 1
    else:
        seq_len = size[0]
        label_len = size[1]
        pred_len = size[2]
    # steps = pred_len
    # init
    assert flag in ['train', 'test', 'val']
    type_map = {'train': 0, 'val': 1, 'test': 2}
    set_type = type_map[flag]

    features = features
    target = target
    scale = scale
    if scale:
        scaler = StandardScaler()
    timeenc = timeenc
    freq = freq

    market_name = market_name
    tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    root_path = root_path
    data_path = data_path
    tickers = np.genfromtxt(
        os.path.join(root_path, data_path, '..', tickers_fname),
        dtype=str, delimiter='\t', skip_header=False
    )
    print('#tickers selected:', len(tickers))

    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    data_time_stamp = []

    if market_name == 'TSE':
        trade_dates = 1188
        valid_index = 693
        test_index = 924
    else:
        trade_dates = 1245
        valid_index = 756
        test_index = 1008
    border1s = [0, valid_index - seq_len, test_index - seq_len]
    border2s = [valid_index, test_index, trade_dates]
    border1 = border1s[set_type]
    border2 = border2s[set_type]

    # read tickers' eod data
    length = len(tickers)
    # length = 200
    for index, ticker in enumerate(tickers):
        if index >= length:
            break
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        df_raw = pd.read_csv(os.path.join(root_path, data_path,
                                          market_name + '_' + ticker + '_1.csv'))

        cols = list(df_raw.columns)
        cols.remove(target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [target]]
        # print(cols)
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            df_raw = df_raw[:-1]
        if features == 'M' or features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif features == 'S':
            df_data = df_raw[[target]]
        data = df_data.values
        # if scale:
        #     train_data = data[border1s[0]:border2s[0]]
        #     scaler.fit(train_data)
        #     data = scaler.transform(data)

        data = data[border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
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
    return eod_data


def create_edge_index(data, k, graph_type):
    # data: channel x nodes x length
    # for b in
    num_nodes = data.shape[1]
    channel = data.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for m in tqdm(range(channel), desc="Processing channel"):
        data_x = data[m]
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            distances = []
            for j in range(num_nodes):
                if i != j:
                    dist, _ = fastdtw(data_x[i].reshape(1, -1), data_x[j].reshape(1, -1), dist=euclidean)
                    distances.append((dist, j))
            distances.sort(key=lambda x: x[0])
            if graph_type == 0:
                adjacency_matrix[i, i] = 1
                for j in range(k-1):
                    adjacency_matrix[distances[j][1], i] = 1
            elif graph_type == 1:
                for j in range(k):
                    adjacency_matrix[i, distances[j][1]] = 1
                    adjacency_matrix[distances[j][1], i] = 1
        if graph_type == 0:  # hypergraph
            if m == 0:
                adj_matrix = adjacency_matrix
            else:
                adj_matrix = np.concatenate((adj_matrix, adjacency_matrix), axis=1)
        else:  # normal graph
            if m == 0:
                adj_matrix = adjacency_matrix
            else:
                adj_matrix = np.sum((adj_matrix, adjacency_matrix), axis=0)
            # adj_matrix_list.append(adjacency_matrix.tolist())

    return adj_matrix


root_path_name="/projects/STGMamba/dataset/NASDAQ_NYSE"
data_path_name="stock_sup/2013-01-01"
K = 10

data = get_data(root_path=root_path_name, data_path=data_path_name, market_name='TSE', flag='train', size=None,
                                   features='MS', target='Close', scale=True, timeenc=0, freq='d')
data = data.transpose(2, 0, 1)
hyperedge_index = create_edge_index(data, K, 0)  # num_stocks, length, features
np.save('/projects/STGMamba/dataset/NASDAQ_NYSE/stock_sup/relation/dtw/TSE_dtw_relation_train_top10_mix.npy', hyperedge_index)
print(hyperedge_index.shape)



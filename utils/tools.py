import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import math
from random import shuffle, choice
import subprocess

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import ndcg_score
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

plt.switch_backend('agg')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, code=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, code)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, code)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, code):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if code == None:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), path + '/' + code + '_checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def get_gpu_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    lines = output.split('\n')

    for line in lines:
        if 'MiB /' in line:
            # print(line)

            gpu_usage = line.split('MiB / ')[0]
            gpu_usage = gpu_usage.split('|')[-1]
            gpu_usage = gpu_usage.strip()

            return (int(gpu_usage) - 10) / 1000

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# loss function
def asymmetric_mse_loss(input, target, beta=0.7):
    """
    Asymmetric MSE Loss Function
    :param input: Predicted values (tensor)
    :param target: Ground truth values (tensor)
    :param beta: Weight for underestimation errors
    :return: Asymmetric MSE loss value (tensor)
    """
    error = input - target
    overestimation = torch.clamp(error, min=0.0)
    underestimation = torch.clamp(-error, min=0.0)

    loss = beta * (overestimation ** 2) + (1 - beta) * (underestimation ** 2)
    return torch.mean(loss)

class TopkMLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, output_dim=100):  # 保持输入输出维度与M一致
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, N, M)
        out = self.mlp(x)
        return out

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def trr_loss_mse_rank(pred, true, base_price, ground_truth, mask, alpha, device):
    # model = TopkMLP(input_dim=1, output_dim=1)
    # model.load_state_dict(torch.load("/projects/STHMamba/models/pretrain/checkpoint.pth"))
    # for name, value in model.named_parameters():
    #     value.requires_grad = False
    # out = torch.mean(model(pred) * ground_truth )

    num_stocks = pred.shape[1]
    length = pred.shape[2]
    return_ratio = torch.div((pred - base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)

    all_ones = torch.ones(pred.size(0), num_stocks, length)
    pre_pw_dif = (torch.matmul(return_ratio, torch.transpose(all_ones, 1, 2))
                  - torch.matmul(all_ones, torch.transpose(return_ratio, 1, 2)))
    gt_pw_dif = (torch.matmul(all_ones, torch.transpose(ground_truth, 1, 2))
                 - torch.matmul(ground_truth, torch.transpose(all_ones, 1, 2)))
    mask_pw = torch.matmul(mask, torch.transpose(mask, 1, 2))
    rank_loss = torch.mean(F.relu(((pre_pw_dif * gt_pw_dif) * mask_pw)))

    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio

def create_edge_index(data, k, graph_type):
    # data: channel x nodes x length
    # for b in
    num_nodes = data.shape[1]
    channel = data.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix_list = []
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
                    adjacency_matrix[distances[j][1], i] = 1.0 / distances[j][0]
            elif graph_type == 1:
                for j in range(k):
                    adjacency_matrix[i, distances[j][1]] = 1
                    adjacency_matrix[distances[j][1], i] = 1
        if graph_type == 0:  # hypergraph
            if m == 0:
                adj_matrix = adjacency_matrix
            else:
                adj_matrix = np.concatenate((adj_matrix, adjacency_matrix), axis=1)
            # adjacency_matrix = dense_to_sparse(torch.from_numpy(adjacency_matrix))
            # adjacency_matrix = adjacency_matrix[0]
            # adj_matrix_list.append(adjacency_matrix.tolist())
        else:  # normal graph
            if m == 0:
                adj_matrix = adjacency_matrix
            else:
                adj_matrix = np.sum((adj_matrix, adjacency_matrix), axis=0)
            # adj_matrix_list.append(adjacency_matrix.tolist())

    if graph_type == 0:
        adj_matrix = np.unique(adj_matrix, axis=1)
        adj_matrix = dense_to_sparse(torch.from_numpy(adj_matrix))
        adj_matrix = adj_matrix[0].tolist()
    elif graph_type == 1:
        adj_matrix[adj_matrix > 1] = 1
    adj_matrix = torch.LongTensor(adj_matrix).to(self.device)

    return adj_matrix

if __name__ == '__main__':
    alpha = 0.1
    num_stocks = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = torch.randn(32, num_stocks, 5)
    true = torch.randn(32, num_stocks, 5)
    base_price = torch.randn(32, num_stocks, 5)
    ground_truth = torch.randn(32, num_stocks, 5)
    mask = torch.ones(32, num_stocks, 1)
    loss, reg_loss, rank_loss, return_ratio = trr_loss_mse_rank(pred, true, base_price, ground_truth, mask, alpha, device)
    # loss2, reg_loss2, rank_loss2, return_ratio2 = trr_loss_mse_rank2(pred, base_price, ground_truth, mask, alpha, device)
    print(loss, reg_loss, rank_loss)
    # print(loss2, reg_loss2, rank_loss2)



import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


# def calc_ic(pred, label):
#     """
#     计算三维张量的 IC 和 rank IC 平均值。
#
#     Parameters:
#     - pred: 预测值，形状为 (batch_size, seq_len, features)
#     - label: 实际值，形状为 (batch_size, seq_len, features)
#
#     Returns:
#     - mean_ic: 当前 batch 的 IC 平均值（标量）
#     - mean_ric: 当前 batch 的 rank IC 平均值（标量）
#     """
#     # 获取维度信息
#     pred = torch.tensor(pred)
#     label = torch.tensor(label)
#
#
#     # 展平 batch 和时间维度 -> (batch_size * seq_len, features)
#     pred_flattened = pred.view(-1, pred.shape[-1])
#     label_flattened = label.view(-1, label.shape[-1])
#
#     # 存储每个特征的 IC 和 rank IC
#     ic_list = []
#     ric_list = []
#
#     # 按特征维度逐列计算
#     for feature_idx in range(pred_flattened.shape[1]):
#         pred_feature = pred_flattened[:, feature_idx]
#         label_feature = label_flattened[:, feature_idx]
#
#         # 转换为 Pandas DataFrame
#         df = pd.DataFrame({'pred': pred_feature, 'label': label_feature})
#
#         # 计算 Pearson 和 Spearman 相关系数
#         ic = df['pred'].corr(df['label'])
#         ric = df['pred'].corr(df['label'], method='spearman')
#
#         ic_list.append(ic)
#         ric_list.append(ric)
#
#     # 计算当前 batch 的 IC 和 rank IC 的均值
#     mean_ic = np.nanmean(ic_list)  # 忽略 NaN
#     mean_ric = np.nanmean(ric_list)  # 忽略 NaN
#
#     return mean_ic, mean_ric

def evaluate(prediction, ground_truth):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}

    # top5
    bt_long5 = 1.0
    ndcg_score_top5 = 0.0
    sharpe_li5 = []

    num_stocks = prediction.shape[0]
    y_score = np.array([[5, 4, 3, 2, 1]])

    for i in range(prediction.shape[1]):
        # sort index
        rank_gt = np.argsort(ground_truth[:, i])
        # ground truth top 5
        gt_top5 = []
        y_true_ = np.zeros(num_stocks, np.int32)
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if len(gt_top5) < 5:
                gt_top5.append(cur_rank)
            y_true_[cur_rank] = num_stocks - j + 1

        # predict top 5
        rank_pre = np.argsort(prediction[:, i])
        k = 0
        pre_top5 = []
        y_true = np.zeros((1, 5), np.int32)
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if len(pre_top5) < 5:
                pre_top5.append(cur_rank)
                y_true[0][k] = y_true_[cur_rank]
                k += 1
        if len(pre_top5) == 5:
            ndcg_score_top5 += ndcg_score(y_true, y_score)
        else:
            ndcg_score_top5 += 0.0

        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        # 累计收益率计算公式
        bt_long5 *= (1 + real_ret_rat_top5)
        sharpe_li5.append(real_ret_rat_top5)

    performance['btl5'] = bt_long5 - 1
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = (np.mean(sharpe_li5) / np.std(sharpe_li5)) * 15.87  # To annualize
    performance['ndcg_score_top5'] = ndcg_score_top5 / prediction.shape[1]

    res_list = [performance['btl5'], performance['sharpe5'], performance['ndcg_score_top5']]
    return res_list

# def calc_ic(pred, true):
#     """计算Pearson/Spearman相关系数
#     Args:
#         pred: 预测值数组 (np.array)
#         true: 真实值数组 (np.array)
#     Returns:
#         ic: Pearson相关系数
#         ric: Spearman秩相关系数
#     """
#     # 将 3D 数组展平为 1D 数组
#     pred = pred.reshape(pred.shape[0], -1)
#     true = true.reshape(true.shape[0], -1)
#
#     ic_list = []
#     ric_list = []
#     icir_list = []
#     ricir_list = []
#     for i in range(pred.shape[1]):
#         pred_flat = pred[:, i]
#         true_flat = true[:, i]
#         # 将展平后的数组转换为DataFrame
#         df = pd.DataFrame({'pred': pred_flat, 'true': true_flat})
#         df['pred_rank'] = df['pred'].rank()
#         df['true_rank'] = df['true'].rank()
#
#         ic = df['pred'].rolling(window=10).corr(df['true'])
#         ric = df['pred_rank'].rolling(window=10).corr(df['true_rank'])
#
#
#         ic_list.append(ic.mean())
#         ric_list.append(ric.mean())
#
#         # # 计算Pearson相关系数和Spearman秩相关系数
#         # non_rolling_ic = df['pred'].corr(df['true'])  # Pearson相关系数
#         # non_rolling_ric = df['pred'].corr(df['true'], method='spearman')  # Spearman秩相关系数
#
#     return ic_list.mean(), ric_list.mean(), icir_list.mean(), ricir_list.mean()


def calc_ic(pred, true):
    """计算Pearson/Spearman相关系数
    Args:
        pred: 预测值数组 (np.array)
        true: 真实值数组 (np.array)
    Returns:
        ic: Pearson相关系数
        ric: Spearman秩相关系数
    """
    pred = pred.reshape(pred.shape[0], -1)
    true = true.reshape(true.shape[0], -1)

    ic_list = []
    rank_ic_list = []
    for i in range(pred.shape[1]):
        # 每一行是一个批次，计算预测和真实值的皮尔逊相关系数
        batch_pred = pred[:, i]
        batch_true = true[:, i]

        # 计算每个批次的 Pearson 相关系数（IC）
        ic = pd.Series(batch_pred).corr(pd.Series(batch_true))  # 使用 pandas 的 corr() 方法
        ic_list.append(ic)

        # 计算每个批次的 Spearman 排序相关系数（Rank IC）
        rank_ic = pd.Series(batch_pred).corr(pd.Series(batch_true), method='spearman')
        rank_ic_list.append(rank_ic)

    # 将 IC 列表转换为 NumPy 数组
    ic_array = np.array(ic_list)
    rank_ic_array = np.array(rank_ic_list)

    # 计算 ICIR
    ic_mean = np.mean(ic_array)

    rank_ic_mean = np.mean(rank_ic_array)

    return ic_mean, rank_ic_mean


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    ic, ric = calc_ic(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, ic, ric

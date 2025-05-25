from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics_why import metric, evaluate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

class Exp_Stocks_Baseline_why(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stocks_Baseline_why, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        reduction = 'mean'
        if self.args.loss in ['MSE', 'L2']:
            criterion = nn.MSELoss(reduction=reduction)
        elif self.args.loss == 'L1':
            criterion = nn.L1Loss(reduction=reduction)
        else:
            raise NotImplementedError

        if self.args.VPT_mode in [1]:
            criterion = nn.MSELoss(reduction='none')

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, price_batch) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                price_batch = price_batch[:, -self.args.pred_len:, f_dim:]

                return_ratio = torch.div((pred - price_batch), price_batch)
                gt_ratio = torch.div((true - price_batch), price_batch)

                loss = criterion(return_ratio, gt_ratio)

                if self.args.VPT_mode in [1]:
                    loss = loss.mean()

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        training_times = []
        code_list = sorted(os.listdir(self.args.root_path))
        for code in tqdm(code_list, desc="training stocks"):
            self.args.data_path = code
            self.args.code = code.replace('.csv', '')
            print("------> training stock: ", self.args.code)
            training_time = self.train_one_code(setting, self.args.code, model_optim, criterion)
            training_times.append(training_time)

        print(f"Average training time: {np.mean(training_times):.2f} seconds")
        print(f"Total training time: {np.sum(training_times):.2f} seconds")

        folder_path = f'{self.args.results}/{setting}/'
        with open(folder_path + "/result_stock_baseline_test.txt", 'w') as f:
            f.write(setting + "  \n")
            f.write('Average training time:{:.2f} seconds, Total training time:{:.2f} seconds'.format(np.mean(training_times), np.sum(training_times)))
            f.write('\n\n')

    def train_one_code(self, setting, code, model_optim, criterion):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.results, setting, 'best_models')
        if not os.path.exists(path):
            os.makedirs(path)

        total_start_time = time.time()
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, delta=self.args.delta, verbose=True)

        from utils.optim_utils import get_lr_scheduler
        total_steps = len(train_loader) * self.args.train_epochs
        if isinstance(self.args.warmup_steps, float):
            self.args.warmup_steps = int(self.args.warmup_steps * total_steps)
        # scheduler = get_lr_scheduler(model_optim, total_steps=total_steps, args=self.args)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            from utils.tools import get_gpu_usage
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, price_batch) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        price_batch = price_batch[:, -self.args.pred_len:, f_dim:].to(self.device)

                        return_ratio = torch.div((outputs - price_batch), price_batch)
                        gt_ratio = torch.div((batch_y - price_batch), price_batch)

                        loss = criterion(return_ratio, gt_ratio)

                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    price_batch = price_batch[:, -self.args.pred_len:, f_dim:].to(self.device)

                    return_ratio = torch.div((outputs - price_batch), price_batch)
                    gt_ratio = torch.div((batch_y - price_batch), price_batch)

                    loss = criterion(return_ratio, gt_ratio)

                    if self.args.VPT_mode in [1]:
                        self.model.batch_update_state(loss)
                        loss = loss.mean()

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print('GPU:', get_gpu_usage())
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # print(f">>> 'sec/iter': {speed}, 'GPU': {get_gpu_usage()}")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # raise Exception
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path, code=code)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if not self.args.no_lradj:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        total_end_time = time.time()
        training_time = total_end_time - total_start_time

        best_model_path = path + '/' + str(code) + '_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # return self.model
        return training_time


    def test(self, setting, test=0):
        metrics_list = []
        preds = []
        trues = []

        folder_path = f'{self.args.results}/{setting}/'
        code_list = sorted(os.listdir(self.args.root_path))
        for code in tqdm(code_list, desc="testing stocks"):
            # data_path是一个具体的股票代码
            self.args.data_path = code
            self.args.code = code.replace('.csv', '')
            print("------> testing code: ", self.args.code)
            metrics, pred, true = self.test_one_code(setting, test, self.args.code)  # mae, mse, rmse, mape, mspe, rse, ic, ric
            metrics_list.append(metrics)
            preds.append(pred)
            trues.append(true)
        metrics_array = np.array(metrics_list)  # all stocks
        preds = np.array(preds)
        trues = np.array(trues)
        perf = evaluate(preds, trues)

        # mean_metrics_array = np.mean(metrics_array, axis=0)
        icir = metrics_array[:, 6].mean() / metrics_array[:, 6].std()
        ricir = metrics_array[:, 7].mean() / metrics_array[:, 7].std()

        print("==========stock baseline testing metrics===========")
        print('mse:{}, mae:{}, rse:{}, ic:{}, ric:{}, icir:{}, ricir:{}'.format(metrics_array[:, 1].mean(), metrics_array[:, 0].mean(), metrics_array[:, 5].mean(), metrics_array[:, 6].mean(), metrics_array[:, 7].mean(), icir, ricir))
        print('irr5:{}, sharpe5:{}, ndcg@5:{}'.format(perf[0], perf[1], perf[2]))

        with open(folder_path + "/result_stock_baseline_test.txt", 'a') as f:
            f.write("Test metrics  \n")
            f.write('mse:{}, mae:{}, rse:{}, ic:{}, ric:{}, icir:{}, ricir:{}\n'.format(metrics_array[:, 1].mean(), metrics_array[:, 0].mean(), metrics_array[:, 5].mean(), metrics_array[:, 6].mean(), metrics_array[:, 7].mean(), icir, ricir))
            f.write('irr5:{}, sharpe5:{}, ndcg@5:{}'.format(perf[0], perf[1], perf[2]))
            f.write('\n\n')
        np.save(folder_path + '/metrics.npy', metrics_array)

    def test_one_code(self, setting, test=0, code=None):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(os.path.join(self.args.results, setting, 'best_models', str(code) + '_checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # visual_path = f'{self.args.results}/{setting}/{code}/{self.args.visualization}'
        # os.makedirs(visual_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, price_batch) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                price_batch = price_batch[:, -self.args.pred_len:, f_dim:]

                return_ratio = torch.div((outputs - price_batch), price_batch)
                return_ratio = return_ratio.detach().cpu().numpy()
                gt_ratio = torch.div((batch_y - price_batch), price_batch)
                gt_ratio = gt_ratio.detach().cpu().numpy()

                preds.append(return_ratio)
                trues.append(gt_ratio)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'

        mae, mse, rmse, mape, mspe, rse, corr, ic, ric = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, ic:{}, ric:{}'.format(mse, mae, rse, ic, ric))
        metrics_array = [mae, mse, rmse, mape, mspe, rse, ic, ric]

        with open(folder_path + "result_stock_baseline_test.txt", 'a') as f:
            f.write(code + " : ")
            f.write('mse:{}, mae:{}, rse:{}, ic:{}, ric:{}'.format(mse, mae, rse, ic, ric))
            f.write('\n\n')

        if not os.path.exists(folder_path + 'preds_trues/'):
            os.makedirs(folder_path + 'preds_trues/')
        np.save(folder_path + 'preds_trues/' + str(code) + '_pred.npy', preds)
        np.save(folder_path + 'preds_trues/' + str(code) + '_true.npy', trues)

        return metrics_array, preds[:, :, 0].flatten(), trues[:, :, 0].flatten()

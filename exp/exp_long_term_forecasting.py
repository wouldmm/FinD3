import copy

import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from thop import profile
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, trr_loss_mse_rank, create_edge_index
from utils.metrics import metric, evaluate, evaluate_topk
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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
        elif self.args.loss == 'rank':
            criterion = trr_loss_mse_rank
        else:
            raise NotImplementedError

        if self.args.VPT_mode in [1]:
            criterion = nn.MSELoss(reduction='none')


        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_reg_loss = []
        total_rank_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    mask_batch, price_batch, gt_batch) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()
                mask_batch = mask_batch.float()
                price_batch = price_batch.float()
                gt_batch = gt_batch.float()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, graph_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0)[0]
                        else:
                            outputs, graph_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0)
                else:
                    if self.args.output_attention:
                        outputs, graph_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0)[0]
                    else:
                        outputs, graph_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, :, -self.args.pred_len:, f_dim:].to(self.device)
                # outputs = outputs[:, :, -self.args.pred_len:]
                # batch_y = batch_y[:, :, -self.args.pred_len:, f_dim:].squeeze(3).to(self.device)

                pred = outputs
                true = batch_y

                if self.args.loss in ['MSE', 'L2']:
                    return_ratio = torch.div((pred.squeeze(-1).cpu() - price_batch), price_batch)
                    loss = criterion(return_ratio.float(), gt_batch.float())
                    # loss = criterion(outputs, batch_y)
                    total_loss.append(loss.item())

                elif self.args.loss == 'rank':
                    loss, reg_loss, rank_loss, rr = criterion(pred.squeeze(-1).cpu(),
                                                              true.squeeze(-1).cpu(),
                                                              torch.FloatTensor(price_batch),
                                                              torch.FloatTensor(gt_batch),
                                                              torch.FloatTensor(mask_batch),
                                                              self.args.alpha, self.device)
                    loss += self.args.lamb * graph_loss.cpu()

                    total_loss.append(loss.item())
                    total_reg_loss.append(reg_loss.item())
                    total_rank_loss.append(rank_loss.item())

                if self.args.VPT_mode in [1]:
                    loss = loss.mean()
                    total_loss.append(loss.item())


        self.model.train()
        if self.args.loss in ['MSE', 'L2']:
            total_loss = np.average(total_loss)
            return total_loss

        elif self.args.loss == 'rank':
            total_loss = np.average(total_loss)
            total_reg_loss = np.average(total_reg_loss)
            total_rank_loss = np.average(total_rank_loss)
            return total_loss, total_reg_loss, total_rank_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.results, setting)

        total_start_time = time.time()
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

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
            train_reg_loss = []
            train_rank_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    mask_batch, price_batch, gt_batch) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()
                mask_batch = mask_batch.float()
                price_batch = price_batch.float()
                gt_batch = gt_batch.float()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                flops, params = profile(self.model, inputs=(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0), verbose=False)
                print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
                print(f"Parameters: {params / 1e6:.4f} M")
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, graph_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0)[0]
                        else:
                            outputs, graph_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0)
                else:
                    if self.args.output_attention:
                        outputs, graph_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0)[0]
                    else:
                        outputs, graph_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 0)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, :, -self.args.pred_len:, f_dim:].to(self.device)
                # if self.args.VPT_mode in [1]:
                #     self.model.batch_update_state(loss)
                #     loss = loss.mean()
                if self.args.loss in ['MSE', 'L2']:
                    return_ratio = torch.div((outputs.squeeze(-1).cpu() - price_batch), price_batch)
                    loss = criterion(return_ratio.float(), gt_batch.float())
                    # loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                elif self.args.loss == 'rank':
                    loss, reg_loss, rank_loss, rr = criterion(outputs.squeeze(-1).cpu(),
                                                              batch_y.squeeze(-1).cpu(),
                                                              torch.FloatTensor(price_batch),
                                                              torch.FloatTensor(gt_batch),
                                                              torch.FloatTensor(mask_batch),
                                                              self.args.alpha, self.device)
                    loss += self.args.lamb * graph_loss.cpu()

                    train_loss.append(loss.item())
                    train_reg_loss.append(reg_loss.item())
                    train_rank_loss.append(rank_loss.item())

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
            if self.args.loss in['MSE', 'L2']:
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            elif self.args.loss == 'rank':
                train_loss = np.average(train_loss)
                train_reg_loss = np.average(train_reg_loss)
                train_rank_loss = np.average(train_rank_loss)
                # raise Exception
                vali_loss, vali_reg, vali_rank = self.vali(vali_data, vali_loader, criterion)
                test_loss, test_reg, test_rank = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                print("                       | Train reg Loss: {2:.7f} Vali reg Loss: {3:.7f} Test reg Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_reg_loss, vali_reg, test_reg))
                print("                       | Train rank Loss: {2:.7f} Vali rank Loss: {3:.7f} Test rank Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_rank_loss, vali_rank, test_rank))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if not self.args.no_lradj:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        folder_path = f'{self.args.results}/{setting}/'
        with open(folder_path + "/result_long_term_forecast.txt", 'w') as f:
            f.write(setting + "  \n")
            f.write('Total training time:{:.2f} seconds\n'.format(total_training_time))
            f.write('\n\n')

        # return self.model
        return total_training_time

    def test(self, setting, test=0):
        total_start_time = time.time()

        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.results, setting, 'checkpoint.pth')))

        # folder_path = './test_results/' + setting + '/'
        folder_path = f'{self.args.results}/{setting}/{self.args.visualization}'
        os.makedirs(folder_path, exist_ok=True)

        results = []
        preds = []
        trues = []
        masks = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    mask_batch, price_batch, gt_batch) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                mask_batch = mask_batch.float()
                price_batch = price_batch.float()
                gt_batch = gt_batch.float()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 1)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 1)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 1)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, :, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                # if test_data.scale and self.args.inverse:
                #     shape = outputs.shape
                #     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                # pred = outputs
                # true = batch_y
                rr = torch.div((outputs.squeeze(-1) - price_batch), price_batch)
                rr = rr.detach().cpu().numpy()

                mask_batch = mask_batch.detach().cpu().numpy()

                gt_batch = gt_batch.detach().cpu().numpy()

                results.append(outputs)
                preds.append(rr)
                trues.append(gt_batch)
                masks.append(mask_batch)

        total_end_time = time.time()
        total_testing_time = total_end_time - total_start_time

        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = f'{self.args.results}/{setting}/'

        preds_calc = preds[:, :, 0].transpose()
        trues_calc = trues[:, :, 0].transpose()
        masks_calc = masks[:, :, 0].transpose()
        print('test shape:', preds_calc.shape, trues_calc.shape, masks_calc.shape)
        perf = evaluate_topk(preds_calc, trues_calc, masks_calc)  # 返回技术指标[mse,ndcg_score_top5,btl5(累计收益率),sharpe5(夏普比率)]

        print('Test preformance:', perf)
        with open(folder_path + "/result_long_term_forecast.txt", 'a') as f:
            # f.write(setting + "  \n")
            f.write('Total testing time:{:.2f} seconds\n'.format(total_testing_time))
            # f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('Test preformance:{}'.format(perf))
            f.write('\n\n')
            f.close()

            # np.save(folder_path + '/metrics.npy', metrics_array)
            np.save(folder_path + '/pred.npy', preds)
            np.save(folder_path + '/true.npy', trues)

        return perf

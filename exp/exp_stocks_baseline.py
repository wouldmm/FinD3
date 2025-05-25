from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, evaluate, evaluate_baseline
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

def get_code_list(root_path, data_path, market_name):
    # df = pd.read_csv(os.path.join(root_path,
    #                               data_path), index_col=0)
    # code_list = df['code'].unique()
    code_list = np.genfromtxt(
        os.path.join(root_path, data_path, '..', market_name+'_tickers_qualify_dr-0.98_min-5_smooth.csv'),
        dtype=str, delimiter='\t', skip_header=False
    )
    return code_list

class Exp_Stocks_Baseline(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stocks_Baseline, self).__init__(args)

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
        total_reg_loss = []
        total_rank_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    mask_batch, price_batch, gt_batch) in enumerate(vali_loader):
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

                if self.args.loss in ['MSE', 'L2']:
                    return_ratio = torch.div((pred.squeeze(-1) - price_batch), price_batch)
                    loss = criterion(return_ratio.float(), gt_batch.float())
                    # loss = criterion(outputs, batch_y)
                    total_loss.append(loss.item())

                elif self.args.loss == 'rank':
                    loss, reg_loss, rank_loss, rr = criterion(outputs.squeeze(-1),
                                                              batch_y.squeeze(-1),
                                                              torch.FloatTensor(price_batch).to(self.device),
                                                              torch.FloatTensor(gt_batch).to(self.device),
                                                              torch.FloatTensor(mask_batch).to(self.device),
                                                              self.args.alpha, self.device)
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
        training_times = []
        code_list = get_code_list(self.args.root_path, self.args.data_path, self.args.market)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for code in tqdm(code_list, desc="training stocks"):
            self.args.code = code
            print("------> training stock: ", code)
            training_time = self.train_one_code(setting, model_optim, criterion, code)
            training_times.append(training_time)

        print(f"Average training time: {np.mean(training_times):.2f} seconds")
        print(f"Total training time: {np.sum(training_times):.2f} seconds")

        folder_path = f'{self.args.results}/{setting}/'
        with open(folder_path + "/result_stock_baseline_test.txt", 'w') as f:
            f.write(setting + "  \n")
            f.write('Average training time:{:.2f} seconds, Total training time:{:.2f} seconds'.format(np.mean(training_times), np.sum(training_times)))
            f.write('\n\n')

    def train_one_code(self, setting, model_optim, criterion, code):
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
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.VPT_mode in [1]:
                    return_ratio = torch.div((outputs.squeeze(-1).cpu() - price_batch), price_batch)
                    loss = criterion(return_ratio.float(), gt_batch.float())
                    self.model.batch_update_state(loss.unsqueeze(-1).to(self.device))
                    loss = loss.mean()
                    train_loss.append(loss.item())
                elif self.args.loss in ['MSE', 'L2']:
                    return_ratio = torch.div((outputs.squeeze(-1).cpu() - price_batch), price_batch)
                    loss = criterion(return_ratio.float(), gt_batch.float())
                    # loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                elif self.args.loss == 'rank':
                    loss, reg_loss, rank_loss, rr =  criterion(outputs.squeeze(-1),
                                                              batch_y.squeeze(-1),
                                                              torch.FloatTensor(price_batch).to(self.device),
                                                              torch.FloatTensor(gt_batch).to(self.device),
                                                              torch.FloatTensor(mask_batch).to(self.device),
                                                              self.args.alpha, self.device)

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

            if self.args.loss in ['MSE', 'L2']:
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
                print(
                    "                       | Train reg Loss: {2:.7f} Vali reg Loss: {3:.7f} Test reg Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_reg_loss, vali_reg, test_reg))
                print(
                    "                       | Train rank Loss: {2:.7f} Vali rank Loss: {3:.7f} Test rank Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_rank_loss, vali_rank, test_rank))

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
        testing_times = []
        folder_path = f'{self.args.results}/{setting}/'

        metrics_list = []
        preds = []
        trues = []
        masks = []
        code_list = get_code_list(self.args.root_path, self.args.data_path, self.args.market)
        for code in tqdm(code_list, desc="testing stocks"):
            self.args.code = code
            print("------> testing code: ", code)
            metrics, total_time, pred, true, mask = self.test_one_code(setting, test, code)  # mae, mse, rmse, mape, mspe
            metrics_list.append(metrics)
            testing_times.append(total_time)
            preds.append(pred)
            trues.append(true)
            masks.append(mask)

        metrics_array = np.array(metrics_list)
        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        np.save(folder_path + '/pred.npy', preds)
        np.save(folder_path + '/true.npy', trues)
        np.save(folder_path + '/masks.npy', masks)
        print("==========stock baseline testing metrics===========")
        print("stock length", len(code_list))
        print(f"Average testing time: {np.mean(testing_times):.2f} seconds")
        print(f"Total testing time: {np.sum(testing_times):.2f} seconds")
        print("MSE: ", np.mean(metrics_array[:, 1]))
        print("MAE: ", np.mean(metrics_array[:, 0]))

        perf = evaluate(preds, trues, masks)  # 返回技术指标[mse,ndcg_score_top5,btl5(累计收益率),sharpe5(夏普比率)]
        print('Test preformance:', perf)

        with open(folder_path + "/result_stock_baseline_test.txt", 'a') as f:
            # f.write(setting + "  \n")
            f.write('Average testing time:{:.2f} seconds, Total testing time:{:.2f} seconds'.format(np.mean(testing_times), np.sum(testing_times)))
            f.write('mse:{}, mae:{}'.format(np.mean(metrics_array[:, 1]), np.mean(metrics_array[:, 0])))
            f.write('\n\n')
            f.write('Test preformance:{}'.format(perf))
            f.write('\n\n')

    def test_one_code(self, setting, test=0, code=None):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            # print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(os.path.join(self.args.results, setting, 'best_models', code + '_checkpoint.pth')))

        # folder_path = './test_results/' + setting + '/'
        # visual_path = f'{self.args.results}/{setting}/{code}/{self.args.visualization}'
        # os.makedirs(visual_path, exist_ok=True)

        results = []
        preds = []
        trues = []
        masks = []

        total_start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,
                    mask_batch, price_batch, gt_batch) in enumerate(test_loader):
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
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                # if test_data.scale and self.args.inverse:
                #     shape = outputs.shape
                #     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                rr = torch.div((outputs.squeeze(-1) - price_batch), price_batch)
                rr = rr.detach().cpu().numpy()

                mask_batch = mask_batch.detach().cpu().numpy()

                gt_batch = gt_batch.detach().cpu().numpy()

                results.append(outputs)
                preds.append(rr)
                trues.append(gt_batch)
                masks.append(mask_batch)

        total_end_time = time.time()
        total_test_time = total_end_time - total_start_time

        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        metrics_array = np.array([mae, mse, rmse, mape, mspe, total_test_time])
        # print('mse:{}, mae:{}'.format(mse, mae))
        return metrics_array, total_test_time, preds[:, :, 0].flatten(), trues[:, :, 0].flatten(), masks.flatten()


if __name__ == '__main__':
    folder_path = '/projects/STGMamba/results/long_term_forecast_NASDAQ_24_1_PatchTST_NASDAQ_baseline_ftMS_sl24_ll0_pl1_dm16_nh16_el5_lr0.001_lsMSE_bs32_a10_patch8_0'
    preds = np.load(folder_path+'/pred.npy')
    trues = np.load(folder_path+'/true.npy')
    perf = evaluate_baseline(preds, trues)  # 返回技术指标[mse,ndcg_score_top5,btl5(累计收益率),sharpe5(夏普比率)]
    print('Test preformance:', perf)

    with open(folder_path + "/result_stock_baseline_test.txt", 'a') as f:
        # f.write(setting + "  \n")
        f.write('Average testing time:{:.2f} seconds, Total testing time:{:.2f} seconds'.format(0.04, 45.69))
        f.write('mse:{}, mae:{}'.format(0.0007249773518812324, 0.01658822122178824))
        f.write('\n\n')
        f.write('Test preformance:{}'.format(perf))
        f.write('\n\n')
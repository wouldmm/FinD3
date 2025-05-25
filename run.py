import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_stocks_baseline import Exp_Stocks_Baseline
from exp.exp_stocks_baseline_why import Exp_Stocks_Baseline_why
from utils.print_args import print_args
from utils.tools import set_seed
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STHMamba')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='NYSEtest', help='model id')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--model', type=str, default='FinD3',
                        help='model name, options: [DLinear, Autoformer, iTransformer, MambaTS, PatchTST, FinD3_gcn, FinD3]')
    parser.add_argument('--des', type=str, default='test', help='exp description')

    # data loader
    parser.add_argument('--data', type=str, default='NASDAQ', help='dataset type, option:[CS300, CS300_baseline, baseline_why, NASDAQ, NASDAQ_baseline]')
    parser.add_argument('--root_path', type=str, default='./dataset/NASDAQ_NYSE', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='stock_sup/2013-01-01', help='data file, options:[CS300, SZ50, xxxxxx.ss.csv, stock.csv], CS300, SZ50 and xxx.csv for baseline training, stock_sup/2013-01-01 for FinD3 training')
    parser.add_argument('--market', type=str, default='NYSE', help='NASDAQ_NYSE dataset type, option:[NASDAQ, NYSE]')
    parser.add_argument('--code', type=str, default=None, help='stock code')
    parser.add_argument('--scale', type=bool, default=True, help='whether data need scale')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, '
                             'S:univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--visualization', type=str,
                        default='./visualization',
                        help='visualization of model')
    parser.add_argument('--results', type=str,
                        default='./results',
                        help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=120, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    parser.add_argument('--max_pred_len', type=int, default=-1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=5, help='output size')
    parser.add_argument('--d_model', type=int, default=24, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--delta', type=float, default=0, help='early stopping delta')
    parser.add_argument('--learning_rate', type=float, default=0.000065, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='rank', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--no_lradj', action='store_true', )
    parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
    parser.add_argument('--momentum', default=0.95, type=float, help='momentum')
    # update lr scheduler by iter
    parser.add_argument('--lradj_by_iter', action='store_true', )
    parser.add_argument('--warmup_steps', default=0.1, type=float, help='warmup')
    parser.add_argument('--iters_per_epoch', default=None, type=str, help='warmup')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # PatchTST
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride length')

    # MambaTS
    parser.add_argument('--VPT_mode', type=int, default=0,
                        help='variable permutation training mode, 0 for no use, 1 for default')
    parser.add_argument('--ATSP_solver', type=str, default='SA', help='ATSP_solver',
                        choices=['Random', 'SA', 'GD', 'LS', 'LK'])
    parser.add_argument('--use_casual_conv', action='store_true', help='use multiple gpus', default=False)

    # STGMamba & STHMamba
    parser.add_argument('--hidden_dim', type=int, default=48, help='HGAT hidden dim')
    parser.add_argument('--ltedge', type=int, default=1900, help='HGAT latent edges of graph')
    parser.add_argument('--decomp', type=int, default=1, help='use decomp or not')
    parser.add_argument('--kernel_size', type=int, default=15, help='kernel_size in decomp')
    parser.add_argument('--expand', type=int, default=2, help='mamba block expand dim')
    parser.add_argument('--head_type', type=str, default='prediction', help='pred head type')
    parser.add_argument('--use_fast_path', type=bool, default=False, help='mamba use_fast_path')
    # graph
    parser.add_argument('--graph', action='store_true', help='use graph')
    parser.add_argument('--no-graph', action='store_false', dest='graph', help='do not use graph')
    parser.set_defaults(graph=True)
    parser.add_argument('--gum_bais', type=float, default=5.0, help='bais in gumbel softmax')
    parser.add_argument('--share_emb', type=int, default=1, help='share emb if 1 else 0')
    parser.add_argument('--graph_type', type=int, default=0, help='0 for hypergraph, 1 for normal graph')
    parser.add_argument('--k', type=int, default=10, help='top K nodes connect to a hyperedge or top K nodes to build normal edge')
    parser.add_argument('--alpha', type=float, default=9, help='alpha in loss')
    parser.add_argument('--lamb', type=float, default=0.01, help='lambda in loss')
    parser.add_argument('--intra', type=int, default=0, help='for ablation')
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # args.use_gpu = False
    if args.iters_per_epoch is not None:
        args.iters_per_epoch = eval(args.iters_per_epoch)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]



    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        if args.data == 'CS300_baseline' or args.data == 'NASDAQ_baseline':
            Exp = Exp_Stocks_Baseline
        elif args.data == 'baseline_why':
            Exp = Exp_Stocks_Baseline_why
        else:
            Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            set_seed(args.seed)
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_lr{}_ls{}_bs{}_a{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.learning_rate,
                args.loss,
                args.batch_size,
                args.alpha,
                args.des, ii)

            path = os.path.join(args.results, setting)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(args.results, setting, "args.txt"), 'w') as f:
                for arg, value in vars(args).items():
                    f.write(f'{arg} : {value}\n')

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

            exp.train(setting)

            if args.VPT_mode in [1]:
                print(f"resetting ids shuffle...")
                exp.model.reset_ids_shuffle()

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            set_seed(args.seed)
            exp.test(setting, test=1)

            torch.cuda.empty_cache()
    else:
        # ii = args.itr
        # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #     args.task_name,
        #     args.model_id,
        #     args.model,
        #     args.data,
        #     args.features,
        #     args.seq_len,
        #     args.label_len,
        #     args.pred_len,
        #     args.d_model,
        #     args.n_heads,
        #     args.e_layers,
        #     args.d_ff,
        #     args.factor,
        #     args.embed,
        #     args.distil,
        #     args.des, ii)
        # for ii in range(args.itr):
        ii = args.itr - 1
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_lr{}_ls{}_bs{}_a{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.learning_rate,
            args.loss,
            args.batch_size,
            args.alpha,
            args.des, ii)
        exp = Exp(args)  # set experiments

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        set_seed(args.seed)
        exp.test(setting, test=1)

        torch.cuda.empty_cache()


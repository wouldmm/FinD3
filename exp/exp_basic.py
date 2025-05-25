import os
import torch
from thop import profile

from models import Autoformer, DLinear, FEDformer, PatchTST, Crossformer, iTransformer, MambaTS, FourierGNN, FinD3_gcn, \
    FinD3, Transformer
from models.Ablation import FinD3_test

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'PatchTST': PatchTST,
            'Crossformer': Crossformer,
            'iTransformer': iTransformer,
            'Transformer': Transformer,
            'FourierGNN': FourierGNN,
            'MambaTS': MambaTS,
            'FinD3_gcn': FinD3_gcn,
            'FinD3': FinD3,
            'FinD3_test': FinD3_test
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # from torchsummary import summary
        # from torchinfo import summary
        # # torch.Size([16, 720, 321]) torch.Size([16, 768, 321]) torch.Size([16, 720, 4]) torch.Size([16, 768, 4])
        # # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
        # summary(self.model, [(self.args.batch_size, self.args.seq_len, self.args.enc_in),
        #                      (self.args.batch_size, self.args.seq_len, self.args.enc_in),
        #                      (self.args.batch_size, self.args.seq_len, 4),
        #                      (self.args.batch_size, self.args.seq_len, 4)], device='cuda')
        # dummy_input = torch.randn(1, 120, 5).to(self.device)
        # flops, params = profile(self.model, inputs=(dummy_input, None, None, None), verbose=False)
        # print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
        # print(f"Parameters: {params / 1e6:.4f} M")

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import HypergraphConv
from tqdm import tqdm

from layers.revin import RevIN
from layers.mamba_ssm_test.modules.mamba_simple import MambaConv2d, Mamba

from layers.pos_encoding import *
from layers.basics import *
from layers.attention import *
from typing import Callable, Optional

from torch import Tensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class Model(nn.Module):
    """
    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """

    def __init__(self, configs, shared_embedding=True, dropout: float = 0., head_dropout=0,
                 head_type="prediction", individual=False, **kwargs):

        super().__init__()
        # base
        self.n_vars = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        if configs.market == 'NYSE':
            self.num_nodes = 1737
            self.num_edges = 6423
        elif configs.market == 'NASDAQ':
            self.num_nodes = 1026
            self.num_edges = 2212
        elif configs.market == 'TSE':
            self.num_nodes = 95
            self.num_edges = 475
        elif configs.market == 'CS300':
            self.num_nodes = 102
            self.num_edges = 612
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.graph = configs.graph
        self.graph_type = configs.graph_type
        self.gpu = configs.gpu
        self.device = torch.device('cuda:{}'.format(self.gpu))

        self.revin = RevIN(num_features=self.num_nodes*self.n_vars)

        # Mamba
        self.e_layers = configs.e_layers
        self.expand = configs.expand

        assert head_type in ['pretrain', 'prediction', 'regression',], 'head type should be either pretrain, prediction, or regression'

        # series_decomp
        self.decomp = configs.decomp
        self.decomp_module = series_decomp(configs.kernel_size)

        # Backbone
        self.encoder_a = DCSSM(features=self.n_vars, num_nodes=self.num_nodes, seq_len=self.seq_len, d_model=self.d_model, c_inner=self.num_nodes, graph=self.graph,
                               shared_embedding=shared_embedding, dropout=dropout, use_fast_path=configs.use_fast_path, e_layers=self.e_layers, expand=self.expand, **kwargs)
        self.encoder_b = DCSSM(features=self.num_nodes, num_nodes=self.n_vars, seq_len=self.seq_len, d_model=self.d_model, c_inner=self.n_vars, graph=self.graph,
                               shared_embedding=shared_embedding, dropout=dropout, use_fast_path=configs.use_fast_path, e_layers=self.e_layers, expand=self.expand, **kwargs)

        self.act = nn.LeakyReLU(0.01)

        # EHGAT
        if configs.market == 'NASDAQ' or configs.market == 'NYSE':
            self.ehgat = EHA(path=os.path.join(configs.root_path, configs.data_path, '..', 'relation/all', configs.market + '_all_relation_train_mix.npy'),
                             num_stocks=self.num_nodes, features=self.n_vars, d_model=self.d_model, dropout=self.dropout, hidden_dim=configs.hidden_dim,
                             n_heads=configs.n_heads, k=configs.k, patch_len=self.patch_len, stride=self.stride, init_edges=self.num_edges, ltedge=configs.ltedge,
                             use_fast_path=False, share_emb=configs.share_emb, device=self.device, bais=configs.gum_bais)
        elif configs.market == 'CS300':
            self.ehgat = EHA(path='relations/hypergraph_index_cs300.npy',
                             num_stocks=self.num_nodes, features=self.n_vars, d_model=self.d_model, dropout=self.dropout, hidden_dim=configs.hidden_dim,
                             n_heads=configs.n_heads, k=configs.k, patch_len=self.patch_len, stride=self.stride, init_edges=self.num_edges, ltedge=configs.ltedge,
                             use_fast_path=False, share_emb=configs.share_emb, device=self.device, bais=configs.gum_bais)


        # Head
        self.head_type = head_type

        self.head = PredictionHead(individual, self.n_vars, self.d_model, self.num_nodes, self.pred_len, head_dropout)

    def forward(self, x, x_mark, x_dec, x_mark_dec, test_mod=0):
        """
        x: tensor [bs x num_nodes x seq_len x features]
        x_mark: tensor [bs, seq_len, 3]
        hyperedge_index: tensor [2 * hype_edge_nums]
        """
        bs, stocks, seq_len, features = x.shape
        x = rearrange(self.revin(rearrange(x, 'b s l f -> b l (s f)'), "norm"), 'b l (s f) -> b f s l', s=stocks)
        x = x.contiguous()

        # decomposition
        if self.decomp == 1:
            res_init, trend_init = self.decomp_module(x)
            res_init = rearrange(self.encoder_a(res_init), 'b f s l -> b s f l')
            trend_init = self.encoder_b(rearrange(trend_init, 'b f s l -> b s f l'))
            z = res_init + trend_init
        else:
            z_a = rearrange(self.encoder_a(x), 'b f s l -> b s f l')
            z_b = self.encoder_b(rearrange(x, 'b f s l -> b s f l'))
            z = z_a + z_b
        # ehgat
        if self.graph:
            if test_mod == 0:
                z, graph_loss = self.ehgat(z, test_mod)
            else:
                z = self.ehgat(z, test_mod)
        else:
            graph_loss = torch.tensor(0.0, device=x.device, requires_grad=False)

        z = self.head(z)
        z = rearrange(z, 'b s f l -> b l (s f)')
        z = self.revin(z, "denorm")
        z = rearrange(z, 'b l (s f) -> b s l f', s=stocks)

        if test_mod == 0:
            return z, graph_loss
        else:
            return z

class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_nodes, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.num_nodes = num_nodes
        self.flatten = flatten
        head_dim = d_model * n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            self.linears2 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            self.bns = nn.ModuleList()
            for i in range(self.num_nodes):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(head_dim, d_model))
                self.linears2.append(nn.Linear(d_model, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
                self.bns.append(nn.BatchNorm1d(forecast_len))
        else:
            self.linear = nn.Linear(d_model, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x num_node x d_model]
        output: [bs x num_node x forecast_len]
        """
        if self.individual:
            x_out = []
            for i in range(self.num_nodes):
                z = self.linears1[i](x)  # z: [bs x d_model]
                z = self.dropouts[i](z)
                z = self.linears2[i](z)  # z: [bs x forecast_len]
                z = self.bns[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x forecast_len]
        else:
            x = self.linear(x)  # x: [bs x num_node x d_model]
            x = self.dropout(x)

        return x

class DCSSM(nn.Module):
    '''
    features: 输入特征维度
    num_nodes: 输入股票节点数
    seq_len: 初始输入时间维度
    d_model: 输出时间维度
    c_inner: 输出节点数

    input: [bs, features, num_nodes, seq_len]
    output: [bs, features, num_nodes, d_model] or [bs, features, c_inner, d_model]

    目前默认c_inner与num_nodes相同，可支持e_layers>1
    '''
    def __init__(self, features, num_nodes, seq_len, d_model=128, c_inner=102, graph=0, shared_embedding=True,
                  dropout=0., use_fast_path=True, e_layers=5, expand=2, **kwargs):

        super().__init__()
        self.features = features
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.d_model = d_model
        self.c_inner = c_inner
        self.graph = graph
        self.shared_embedding = shared_embedding

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.features): self.W_P.append(nn.Linear(seq_len, d_model))
        else:
            self.W_P = nn.Linear(seq_len, d_model)
        self.linear = nn.Linear(seq_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        # features 代表特征维度，内部会扩展为delta、B、C
        # d_model 代表处理后的时间长度
        # num_nodes为传感器数
        self.mamba_conv2d = nn.ModuleList(
            [MambaConv2d(
                features=self.features,  # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=3,  # Local convolution width
                expand=expand,
                seq_len=self.d_model,
                num_node=self.num_nodes,
                c_inner=c_inner,
                use_fast_path=use_fast_path
            )
                for i in range(e_layers)
            ]
        )
        self.bn1 = nn.BatchNorm1d(self.features)
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(self.d_model) for i in range(e_layers)]
        )

        self.W_pos = positional_encoding('zero', True, num_nodes, d_model)
        self.revin = RevIN(num_features=self.num_nodes * self.features)

    def forward(self, x) -> Tensor:
        """
        x: tensor [bs x channel x nodes x seq_len]
        x_mark: tensor [bs, seq_len, 3]

        out: [bs x c_inner x nodes x d_model]
        """
        bs, features, nodes, seq_len = x.shape
        x = rearrange(self.revin(rearrange(x, 'b f s l -> b l (s f)'), "norm"), 'b l (s f) -> b f s l', s=nodes)

        if not self.shared_embedding:
            x_out = []
            for i in range(features):
                z = self.W_P[i](x[:, i, :, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            # x = self.linear1(x)
            x = self.W_P(x)  # x: [bs x features x nodes x d_model]
        z = self.dropout(x + self.W_pos)  # z: [bs x features x nodes x d_model]

        # Encoder & residual
        z = rearrange(z, 'b f s l -> b s l f')  # z: [bs x nodes x d_model x features]
        for layer, bn in zip(self.mamba_conv2d, self.bns):
            z = layer(z) + z  # z: [bs x nodes x d_model x features]
            z = bn(z.view(-1, self.d_model)).view(bs, self.c_inner, self.d_model, -1)

        z = rearrange(self.revin(rearrange(z, 'b s l f -> b l (s f)'), 'denorm'), 'b l (s f) -> b f s l', s=nodes)
        z = z.contiguous()  # z: [bs x features x nodes x d_model]
        return z  # z: [bs x features x nodes x d_model]

# Cell
class EHA(nn.Module):
    def __init__(self, path, num_stocks, features, d_model, dropout, hidden_dim, n_heads, k, patch_len, stride, init_edges, ltedge, use_fast_path, share_emb, device, bais):
        super().__init__()
        self.path = path
        self.num_stocks = num_stocks
        self.features = features
        self.k = k
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.device = device
        self.init_edges = init_edges
        self.share_emb = share_emb
        self.ltedge = ltedge
        self.bais = bais

        # HGAT
        self.hgat = HGAT(self.patch_len * self.features, dropout, hidden_dim, n_heads)
        self.patch_num = int((self.d_model - self.patch_len) / self.stride + 1)
        self.value_emb = nn.Linear(self.d_model, self.d_model)
        self.W_pos = positional_encoding('zeros', True, self.features, self.d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1d_0 = nn.Conv1d(num_stocks, num_stocks, kernel_size=3, stride=1, padding=1)
        self.index_mamba0 = Mamba(
            d_model=self.init_edges,
            d_state=16,  # SSM state expansion factor
            d_conv=3,  # Local convolution width
            expand=1,
            use_fast_path=use_fast_path
        )
        if self.share_emb == 0:
            self.index_mamba1 = nn.ModuleList([
                Mamba(
                    d_model=self.ltedge,
                    d_state=16,  # SSM state expansion factor
                    d_conv=3,  # Local convolution width
                    expand=1,
                    use_fast_path=use_fast_path
                ) for i in range(self.patch_num - 1)
            ])
            self.conv1d_1 = nn.ModuleList([
                nn.Conv1d(num_stocks, num_stocks, kernel_size=3, stride=1, padding=1) for i in range(self.patch_num - 1)
            ])
        else:
            self.index_mamba1 = Mamba(
                d_model=self.ltedge,
                d_state=16,  # SSM state expansion factor
                d_conv=3,  # Local convolution width
                expand=1,
                use_fast_path=use_fast_path
            )
            self.conv1d_1 = nn.Conv1d(num_stocks, num_stocks, kernel_size=3, stride=1, padding=1)

        self.adapter = nn.AdaptiveAvgPool1d(self.ltedge)
        self.bn = nn.BatchNorm1d(self.features * self.patch_len)

        self.temperature = nn.Parameter(torch.tensor(0.0))  # 可调节软化系数
        self.sigmod = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.index_stocks = self.init_hypergraph()

    def init_hypergraph(self):
        # prior information
        index_stocks = torch.from_numpy(np.load(self.path))
        index_stocks = dense_to_sparse(index_stocks)[0]
        index_stocks = index_stocks.to(self.device)
        return index_stocks

    def merge_hypergraphs(self, sparse_tensor):
        """
        将批量超图合并为大图
        Args:
            sparse_tensor: List of hyperedge_index tensors, length = batch_size * num_features
        Returns:
            merged_hyperedge_index: 合并后的超边索引 [2, total_edges]
            batch_offsets: 各子图超边的偏移量
        """
        hyperedge_indices = []
        node_offset = 0
        edge_offset = 0
        batch_offsets = []

        for b in range(len(sparse_tensor)):
            hyperedge_index = sparse_tensor[b]
            # 计算偏移
            num_nodes = hyperedge_index[0].max().item() + 1
            num_edges = hyperedge_index[1].max().item() + 1

            # 偏移节点和超边索引
            shifted_nodes = hyperedge_index[0] + node_offset
            shifted_edges = hyperedge_index[1] + edge_offset
            hyperedge_indices.append(torch.stack([shifted_nodes, shifted_edges]))

            batch_offsets.append({
                "node_offset": node_offset,
                "edge_offset": edge_offset,
                "num_nodes": num_nodes,
                "num_edges": num_edges
            })
            node_offset += num_nodes
            edge_offset += num_edges

        merged_hyperedge_index = torch.cat(hyperedge_indices, dim=1)
        return merged_hyperedge_index, batch_offsets

    def hypergraph_laplacian(self, H_cont, eps=1e-6, W=None):
        """
        H_cont: [N, M]，超图 incidence 矩阵
        eps: 防止除零的参数
        W: 超边权重矩阵，如果为None，使用单位对角矩阵
        """
        N, M = H_cont.shape
        if W is None:
            W = torch.eye(M, device=H_cont.device)

        D_v = torch.diag((H_cont @ W @ torch.ones(M, device=H_cont.device)) + eps)  # 节点度矩阵 D_v
        D_e = torch.diag((H_cont.t() @ torch.ones(N, device=H_cont.device)) + eps)  # 超边度矩阵 D_e

        D_v_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D_v)))  # D_v^{-1/2}
        D_e_inv = torch.diag(1.0 / torch.diag(D_e))  # D_e^{-1}

        # 超图拉普拉斯
        I = torch.eye(N, device=H_cont.device)
        L_hyp = I - D_v_inv_sqrt @ H_cont @ W @ D_e_inv @ H_cont.t() @ D_v_inv_sqrt
        return L_hyp

    def hypergraph_laplacian_loss(self, H_cont, node_embeddings):
        """
        H_cont: [B, N, M] 批量超图关联矩阵
        node_embeddings: [B, N, F] 节点嵌入
        """
        loss = 0.0
        for i in range(H_cont.shape[0]):
            L_hyp = self.hypergraph_laplacian(H_cont[i])  # [N, N]
            # Laplacian loss: log(1 + trace(Z^T L Z))
            loss += torch.log(1 + torch.trace(node_embeddings[i].t() @ L_hyp @ node_embeddings[i]))
        loss = loss / H_cont.shape[0]
        return loss

    def forward(self, x, test_mod=0):
        '''
        input: bs x stocks x features x d_model
        output: bs x stocks x features x d_model
        '''
        bs, stocks, features, d_model = x.shape
        tau = torch.exp(self.temperature)
        x = x + self.W_pos
        x = self.dropout(x)
        # do patching
        x = x.unfold(dimension=-1, size=self.patch_len,
                     step=self.stride)  # z_a: tensor [bs x stocks x features x patch_num x patch_len]
        x = rearrange(x, 'b s f p l -> b s p (f l)')

        # step
        sparse_tensor = self.index_stocks.unsqueeze(0).repeat(bs, 1, 1)
        z_out = []
        graph_loss = torch.tensor(0.0, device=x.device, requires_grad=False)
        for step in range(x.shape[2]):
            merged_hyperedge_index, batch_offsets = self.merge_hypergraphs(sparse_tensor)
            if step == 0:
                num_edges = merged_hyperedge_index[1].max().item() + 1
                hyperedge_weight = torch.ones(num_edges).to(self.device)
            x_step = x[:, :, step, :].view(-1, self.patch_len * features)  # [bs*stocks, features * patch_len]
            hgat_out = self.hgat(x_step, merged_hyperedge_index, hyperedge_weight, device=self.device)
            hgat_out = hgat_out.view(bs, stocks, -1)
            z_out.append(hgat_out)

            if x.shape[2] != 1:
                # 处理dense
                dense = []
                for b in range(bs):
                    if step == 0:
                        edges = self.init_edges
                    else:
                        edges = self.ltedge
                    a = to_dense_adj(sparse_tensor[b])[0]
                    if stocks > edges:
                        a = a[:, :edges]
                    else:
                        a = a[:stocks]
                    dense.append(a)
                dense = torch.stack(dense, dim=0)  # b i e
                # dense 演化
                if step == 0:
                    dense = self.index_mamba0(dense)
                    dense = self.conv1d_0(dense)
                    dense = self.adapter(dense)
                else:
                    if self.share_emb == 0:
                        dense = self.index_mamba1[step - 1](dense)
                    else:
                        dense = self.index_mamba1(dense)

                dense = self.relu(dense)
                hyperedge_weight = torch.mean(dense, dim=-2).flatten()

                logits = torch.stack([torch.zeros_like(dense), dense - self.bais], dim=-1)
                result = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
                result = result[..., 1]  # soft二值化
                # print(result.sum(dim=1))
                dense_mod = dense.detach().clone()
                if test_mod == 0:
                    graph_loss += self.hypergraph_laplacian_loss(result, hgat_out)

                result_mod = result.detach().clone()
                row_sums = result_mod.sum(dim=2)  # [B, N]
                zero_rows = (row_sums == 0)  # [B, N]
                max_edge_indices = dense_mod.argmax(dim=2)  # [B, N]
                result_mod[zero_rows, max_edge_indices[zero_rows]] = 1.0

                col_sums = result_mod.sum(dim=1)  # [B, N]
                zero_cols = (col_sums == 0)  # [B, N]
                max_node_indices = dense_mod.argmax(dim=1)  # [B, N]
                batch_idx, edge_idx = torch.where(zero_cols)
                node_idx = max_node_indices[batch_idx, edge_idx]
                result_mod[batch_idx, node_idx, edge_idx] = 1.0
                sparse_tensor = [dense_to_sparse(result_mod[b])[0] for b in range(bs)]
        graph_loss /= x.shape[2]

        z = torch.stack(z_out, dim=2)  # bs x stocks x patch_num x (features x patch_length)
        z = self.bn(z.reshape(-1, z.size(-1))).reshape(bs, stocks, self.patch_num, features, -1)
        z = rearrange(z, 'b s p f l -> b s f (p l)', f=features)

        if test_mod == 0:
            return z, graph_loss
        else:
            return z

class HGAT(nn.Module):
    def __init__(self, d_model, dropout, hidden_dim, n_heads):
        super().__init__()
        self.d_model = d_model

        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.hatt1 = HypergraphConv(hidden_dim, hidden_dim, use_attention=True, heads=n_heads, concat=False, negative_slope=0.2,
                                    dropout=0.2, bias=True)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.hatt2 = HypergraphConv(hidden_dim, hidden_dim, use_attention=True, heads=1, concat=False, negative_slope=0.2,
                                    dropout=0.2, bias=True)
        self.dropout3 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.bn4 = nn.BatchNorm1d(d_model)

    def forward(self, x, hyperedge_index, hyperedge_weight,
                device):  # x: [num_node x d_model] hyperedge_index: [2 x hyperedge_nums]
        self.device = device
        src = x  # src: [num_node x d_model]

        x = F.leaky_relu(self.linear1(x), 0.2) #  x: [num_node x hidden_dim]
        x = self.dropout1(x)
        x = self.bn1(x)  # x: [num_node x hidden_dim]

        # num_nodes = x.shape[1]
        num_nodes = x.shape[0]
        num_edges = hyperedge_index[1].max().item() + 1

        a = to_dense_adj(hyperedge_index)[0].to(self.device)  # a: [num_node x num_edges]
        if num_nodes > num_edges:
            a = a[:, :num_edges]
        else:
            a = a[:num_nodes]
        hyperedge_attr = torch.matmul(a.T, x)

        x2 = self.hatt1(x, hyperedge_index, hyperedge_weight, hyperedge_attr)  # x2: [batch_size, num_nodes, hidden_dim]
        # Add & Norm
        x = x + self.dropout2(x2)  # Add: residual connection with residual dropout
        x = self.bn2(x)

        hyperedge_attr = torch.matmul(a.T, x)
        x2 = self.hatt2(x, hyperedge_index, hyperedge_weight, hyperedge_attr)  # x2: [num_nodes, hidden_dim]
        # Add & Norm
        x = x + self.dropout3(x2)  # Add: residual connection with residual dropout
        x = self.bn3(x)

        x = F.leaky_relu(self.linear2(x), 0.2)  # x: [num_nodes * d_model]
        # Add & Norm
        x = src + self.dropout4(x)  # Add: residual connection with residual dropout
        x = x.contiguous()
        x = self.bn4(x)  # x: [num_nodes, d_model]

        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (batch, stocks, features, length)
        batch, stocks, features, length = x.size()

        # Reshape to (batch * stocks * features, 1, length)
        x = x.view(-1, 1, length)

        # Padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)

        # Apply moving average
        x = self.avg(x)

        # Reshape back to (batch, stocks, features, length)
        x = x.view(batch, stocks, features, -1)

        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # x: (batch, stocks, features, length)
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


if __name__ == '__main__':
    torch.manual_seed(42)

    batch_size = 3
    num_nodes = 5
    num_edges = 4

    dense = torch.randn(batch_size, num_nodes, num_edges)
    print(dense)
    top_k_indices = torch.topk(dense, 2, dim=1).indices  # [bs, k, num_edges]
    result = torch.zeros_like(dense)
    result.scatter_(1, top_k_indices, 1)

    print("\n转换后的密集邻接矩阵 (dense_adj):")
    print(result)

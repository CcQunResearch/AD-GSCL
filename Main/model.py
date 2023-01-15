# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 20:23
# @Author  :
# @Email   :
# @File    : model.py
# @Software: PyCharm
# @Note    :
from functools import partial
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, GRU
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Parameter
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import global_mean_pool, global_add_pool, NNConv, Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
import random
import copy
from Main.infomax import *


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_norm (bool, optional): whether or not to normalize adj matrix.
            (default: :obj:`True`)
        gfn (bool, optional): If `True`, only linear transform (1x1 conv) is
            applied to every nodes. (default: :obj:`False`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 edge_norm=True,
                 gfn=False):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes,),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        edge_index = edge_index[0]
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ResGCN(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, dataset=None, t=0.3, num_classes=2, hidden=128, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True):
        super(ResGCN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.num_classes = num_classes
        self.t = t
        self.nrlabel = 0 if num_classes == 2 else 3
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        self.use_xg = False
        if "xg" in dataset[0]:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)

        hidden_in = dataset.num_features
        if collapse:
            self.bn_feat = BatchNorm1d(hidden_in)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden_in, hidden_in),
                    torch.nn.ReLU(),
                    Linear(hidden_in, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden))
                hidden_in = hidden
            self.lin_class = Linear(hidden_in, self.num_classes)
        else:
            self.bn_feat = BatchNorm1d(hidden_in)
            feat_gfn = True  # set true so GCNConv is feat transform
            self.conv_feat = GCNConv(hidden_in, hidden, gfn=feat_gfn)
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    Linear(hidden, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            if self.res_branch == "resnet":
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=feat_gfn))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GConv(hidden, hidden))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=feat_gfn))
            else:
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GConv(hidden, hidden))
            self.bn_hidden = BatchNorm1d(hidden)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden))
                self.lins.append(Linear(hidden, hidden))
            self.lin_class = Linear(hidden, self.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.collapse:
            return self.forward_collapse(x, edge_index, batch, xg)
        elif self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(x, edge_index, batch, xg)
        elif self.res_branch == "BNReLUConv":
            return self.forward_BNReLUConv(x, edge_index, batch, xg)
        elif self.res_branch == "ConvReLUBN":
            return self.forward_ConvReLUBN(x, edge_index, batch, xg)
        elif self.res_branch == "resnet":
            return self.forward_resnet(x, edge_index, batch, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_collapse(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_BNConvReLU(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_BNReLUConv(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(self.bns_conv[i](x))
            x_ = conv(x_, edge_index)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_ if self.fc_residual else x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_ConvReLUBN(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        x = self.bn_hidden(x)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(conv(x, edge_index))
            x_ = self.bns_conv[i](x_)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(lin(x))
            x_ = self.bns_fc[i](x_)
            x = x + x_ if self.fc_residual else x_
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_resnet(self, x, edge_index, batch, xg=None):
        # this mimics resnet architecture in cv.
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        for i in range(len(self.convs) // 3):
            x_ = x
            x_ = F.relu(self.bns_conv[i * 3 + 0](x_))
            x_ = self.convs[i * 3 + 0](x_, edge_index)
            x_ = F.relu(self.bns_conv[i * 3 + 1](x_))
            x_ = self.convs[i * 3 + 1](x_, edge_index)
            x_ = F.relu(self.bns_conv[i * 3 + 2](x_))
            x_ = self.convs[i * 3 + 2](x_, edge_index)
            x = x + x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class ResGCN_graphcl(ResGCN):
    def __init__(self, **kargs):
        super(ResGCN_graphcl, self).__init__(**kargs)
        hidden = kargs['hidden']
        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.local_d = FF(hidden, hidden)

    def forward_graphrep(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x

    def contrastive_loss(self, x, y, device):
        dot_matrix = torch.mm(x, x.t())  # [bs, bs]
        x_norm = torch.norm(x, p=2, dim=1)  # [bs]
        x_norm = x_norm.unsqueeze(1)  # [bs, 1])
        norm_matrix = torch.mm(x_norm, x_norm.t())  # [bs, bs]

        cos_matrix = (dot_matrix / norm_matrix) / self.t  # [bs, bs]
        cos_matrix = torch.exp(cos_matrix)  # [bs, bs]
        cos_matrix = cos_matrix - torch.diag_embed(torch.diag(cos_matrix))  # [bs, bs]

        y_matrix = y.expand(len(y), len(y))  # [bs, bs]
        neg_indices = torch.ne(y_matrix.t(), y_matrix).float()  # [bs, bs]
        neg_matrix = cos_matrix * neg_indices  # [bs, bs]
        sum_neg_vector = (torch.sum(neg_matrix, dim=1)).unsqueeze(1)  # [bs, 1]

        y_matrix_mask_nrlabel = torch.where(y_matrix != self.nrlabel, y_matrix,
                                            torch.ones(len(y), len(y)).to(device) * 100).to(device)  # [bs, bs]
        pos_indices = torch.eq(y_matrix.t(), y_matrix_mask_nrlabel).float()  # [bs, bs]
        pos_matrix = cos_matrix * pos_indices  # [bs, bs]

        nrmask = torch.where(y != self.nrlabel, torch.ones(len(y)).bool().to(device),
                             torch.zeros(len(y)).bool().to(device)).to(device)  # [bs]

        div = pos_matrix / sum_neg_vector  # [bs, bs]
        div = (torch.sum(div, dim=1)[nrmask]).unsqueeze(1)  # [bs, 1]
        div = div / len(y)  # [bs, 1]
        loss = -torch.sum(torch.log(div))
        return loss

    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)

        feat_map = x

        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x, feat_map

    def infomax_loss(self, data):
        y, M = self.encode(data)
        g_enc = y
        l_enc = self.local_d(M)

        measure = 'JSD'
        loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.batch, measure)
        return loss


######################################################


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, tddroprate):
        super(TDrumorGCN, self).__init__()
        self.tddroprate = tddroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x, edge_index = data.x, data.edge_index

        edge_index_list = edge_index.tolist()
        if self.tddroprate > 0:
            length = len(edge_index_list[0])
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            tdrow = list(np.array(edge_index_list[0])[poslist])
            tdcol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([tdrow, tdcol]).to(device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, budroprate):
        super(BUrumorGCN, self).__init__()
        self.budroprate = budroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x = data.x
        edge_index = data.edge_index.clone()
        edge_index[0], edge_index[1] = data.edge_index[1], data.edge_index[0]

        edge_index_list = edge_index.tolist()
        if self.budroprate > 0:
            length = len(edge_index_list[0])
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            burow = list(np.array(edge_index_list[0])[poslist])
            bucol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([burow, bucol]).to(device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BiGCN_graphcl(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, t, num_classes, tddroprate=0.0, budroprate=0.0):
        super(BiGCN_graphcl, self).__init__()
        self.t = t
        self.nrlabel = 0 if num_classes == 2 else 3
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, tddroprate)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, budroprate)
        self.proj_head = torch.nn.Linear((out_feats + hid_feats) * 2, out_feats)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_classes)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def forward_graphrep(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.proj_head(x)
        return x

    def contrastive_loss(self, x, y, device):
        dot_matrix = torch.mm(x, x.t())  # [bs, bs]
        x_norm = torch.norm(x, p=2, dim=1)  # [bs]
        x_norm = x_norm.unsqueeze(1)  # [bs, 1])
        norm_matrix = torch.mm(x_norm, x_norm.t())  # [bs, bs]

        cos_matrix = (dot_matrix / norm_matrix) / self.t  # [bs, bs]
        cos_matrix = torch.exp(cos_matrix)  # [bs, bs]
        cos_matrix = cos_matrix - torch.diag_embed(torch.diag(cos_matrix))  # [bs, bs]

        y_matrix = y.expand(len(y), len(y))  # [bs, bs]
        neg_indices = torch.ne(y_matrix.t(), y_matrix).float()  # [bs, bs]
        neg_matrix = cos_matrix * neg_indices  # [bs, bs]
        sum_neg_vector = (torch.sum(neg_matrix, dim=1)).unsqueeze(1)  # [bs, 1]

        y_matrix_mask_nrlabel = torch.where(y_matrix != self.nrlabel, y_matrix,
                                            torch.ones(len(y), len(y)).to(device) * 100).to(device)  # [bs, bs]
        pos_indices = torch.eq(y_matrix.t(), y_matrix_mask_nrlabel).float()  # [bs, bs]
        pos_matrix = cos_matrix * pos_indices  # [bs, bs]

        nrmask = torch.where(y != self.nrlabel, torch.ones(len(y)).bool().to(device),
                             torch.zeros(len(y)).bool().to(device)).to(device)  # [bs]

        div = pos_matrix / sum_neg_vector  # [bs, bs]
        div = (torch.sum(div, dim=1)[nrmask]).unsqueeze(1)  # [bs, 1]
        div = div / len(y)  # [bs, 1]
        loss = -torch.sum(torch.log(div))
        return loss


######################################################

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(Encoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(1, dim), ReLU(), Linear(dim, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        feat_map = None
        for i in range(3):
            edge_attr = torch.ones(data.edge_index.shape[1], 1).to(data.x.device)
            m = F.relu(self.conv(out, data.edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map = out
        out = self.set2set(out, data.batch)
        # out = scatter_mean(out, data.batch, dim=0)
        return out, feat_map


class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class InfoMax_graphcl(torch.nn.Module):
    def __init__(self, num_features, dim, t, num_classes):
        super(InfoMax_graphcl, self).__init__()

        self.t = t
        self.nrlabel = 0 if num_classes == 2 else 3

        self.embedding_dim = dim
        self.encoder = Encoder(num_features, dim)

        self.fc1 = torch.nn.Linear(2 * dim, dim)
        # self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, num_classes)

        self.local_d = FF(dim, dim)
        self.global_d = FF(2 * dim, dim)
        # self.global_d = FF(dim, dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        out, M = self.encoder(data)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out, dim=-1)

    def forward_graphrep(self, data):
        out, M = self.encoder(data)
        return out

    def contrastive_loss(self, x, y, device):
        dot_matrix = torch.mm(x, x.t())  # [bs, bs]
        x_norm = torch.norm(x, p=2, dim=1)  # [bs]
        x_norm = x_norm.unsqueeze(1)  # [bs, 1])
        norm_matrix = torch.mm(x_norm, x_norm.t())  # [bs, bs]

        cos_matrix = (dot_matrix / norm_matrix) / self.t  # [bs, bs]
        cos_matrix = torch.exp(cos_matrix)  # [bs, bs]
        cos_matrix = cos_matrix - torch.diag_embed(torch.diag(cos_matrix))  # [bs, bs]

        y_matrix = y.expand(len(y), len(y))  # [bs, bs]
        neg_indices = torch.ne(y_matrix.t(), y_matrix).float()  # [bs, bs]
        neg_matrix = cos_matrix * neg_indices  # [bs, bs]
        sum_neg_vector = (torch.sum(neg_matrix, dim=1)).unsqueeze(1)  # [bs, 1]

        y_matrix_mask_nrlabel = torch.where(y_matrix != self.nrlabel, y_matrix,
                                            torch.ones(len(y), len(y)).to(device) * 100).to(device)  # [bs, bs]
        pos_indices = torch.eq(y_matrix.t(), y_matrix_mask_nrlabel).float()  # [bs, bs]
        pos_matrix = cos_matrix * pos_indices  # [bs, bs]

        nrmask = torch.where(y != self.nrlabel, torch.ones(len(y)).bool().to(device),
                             torch.zeros(len(y)).bool().to(device)).to(device)  # [bs]

        div = pos_matrix / sum_neg_vector  # [bs, bs]
        div = (torch.sum(div, dim=1)[nrmask]).unsqueeze(1)  # [bs, 1]
        div = div / len(y)  # [bs, 1]
        loss = -torch.sum(torch.log(div))
        return loss

    def infomax_loss(self, data):
        y, M = self.encoder(data)
        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        measure = 'JSD'
        loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.batch, measure)
        return loss

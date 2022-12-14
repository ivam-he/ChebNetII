import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm
from torch.nn import Parameter, Linear
from ChebnetII_pro import ChebnetII_prop

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,dropout=.5, is_bns=True):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.is_bns = is_bns
        if is_bns:
            self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if is_bns:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if is_bns:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.is_bns:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        if self.is_bns:
            for i, lin in enumerate(self.lins[:-1]):
                x = lin(x)
                x = F.relu(x, inplace=True)
                x = self.bns[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
            return x
        else:
            for i, lin in enumerate(self.lins[:-1]):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = lin(x)
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
            return x

class ChebNetII(torch.nn.Module):
    def __init__(self,num_features,num_classes, args):
        super(ChebNetII, self).__init__()
        self.name = args.dataset
        self.mlp = MLP(num_features, args.hidden_channels, num_classes, args.num_layers, args.dropout, is_bns=args.is_bns)
        self.prop1 = ChebnetII_prop(args.K, self.name)
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x=self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        
        return x

class ChebNetII_V(torch.nn.Module):
    def __init__(self,num_features,num_classes, args):
        super(ChebNetII_V, self).__init__()
        self.name = args.dataset
        self.mlp = MLP(num_features, args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout, is_bns=args.is_bns)
        self.mlp1 = MLP(args.hidden_channels, args.hidden_channels, num_classes, args.num_layers, args.dropout, is_bns=args.is_bns)
        self.prop1 = ChebnetII_prop(args.K, self.name)
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.mlp.reset_parameters()
        self.mlp1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.graph['node_feat'],data.graph['edge_index']
        x=self.mlp(data)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)

        x=self.mlp1(x,input_tensor=True)
        return x


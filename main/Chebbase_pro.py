import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops,get_laplacian
import torch.nn.functional as F
import numpy as np

class Chebbase_prop(MessagePassing):
    def __init__(self, K, q,bias=True, **kwargs):
        super(Chebbase_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()
        self.q=q #The positive constant

    def reset_parameters(self):
        self.temp.data.fill_(0.0)
        self.temp.data[0]=1.0

    def forward(self, x, edge_index,edge_weight=None):
        coe=self.temp

        #L
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #L_tilde=L-I
        edge_index_tilde, norm_tilde= add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(self.node_dim))

        Tx_0=x
        Tx_1=self.propagate(edge_index_tilde,x=x,norm=norm_tilde,size=None)

        out=coe[0]*Tx_0+coe[1]*Tx_1

        for i in range(2,self.K+1):
            Tx_2=self.propagate(edge_index_tilde,x=Tx_1,norm=norm_tilde,size=None)
            Tx_2=2*Tx_2-Tx_0
            out=out+(coe[i]/i**self.q)*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2
        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch.nn import Linear
from utils import cheby

class ChebNetII(torch.nn.Module):
    def __init__(self,num_features,hidden,num_classes, args):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.lin3 = Linear(hidden, num_classes)
        
        self.K = args.K
        self.temp = Parameter(torch.Tensor(self.K+1))

        #self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x_lis,st=0,end=0):
        coe_tmp=F.relu(self.temp)
        coe=coe_tmp.clone()
        
        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)

        Tx_0=x_lis[0][st:end,:]
        out=coe[0]/2*Tx_0

        for k in range(1,self.K+1):
            Tx_k=x_lis[k][st:end,:]
            out = out+coe[k]*Tx_k

        x = self.lin1(out)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training) 

        x = self.lin3(x)

        return F.log_softmax(x, dim=1)
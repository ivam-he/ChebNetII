import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
import uuid
import math
import os.path as osp
import torch_geometric.transforms as T
from torch.nn import Parameter
import scipy.sparse as sp
from utils import cheby, set_seed, accuracy
from dataset import load_nc_dataset
from data_utils import load_fixed_splits

def create_batch(input_data):
    num_sample = input_data[0].shape[0]
    list_bat = []
    for i in range(0,num_sample,batch_size):
        if (i+batch_size)<num_sample:
            list_bat.append((i,i+batch_size))
        else:
            list_bat.append((i,num_sample))
    return list_bat

def train(st,end):
    model.train()
    optimizer.zero_grad()
    x_lis=[i[st:end,:].to(device) for i in train_data]
    output = model(x_lis)
    acc_train = accuracy(output, train_labels[st:end])
    #print(output)
    #print(output.shape)
    #print(train_labels[st:end].shape)
    loss_train = F.nll_loss(output, train_labels[st:end])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()

def validate(st,end):
    model.eval()
    with torch.no_grad():
        x_lis=[i[st:end,:].to(device) for i in valid_data]
        output = model(x_lis)
        loss_val = F.nll_loss(output, valid_labels[st:end])
        acc_val = accuracy(output, valid_labels[st:end],batch=True)
        return loss_val.item(),acc_val.item()

def test(st,end):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        x_lis=[i[st:end,:].to(device) for i in test_data]
        output = model(x_lis)
        loss_test = F.nll_loss(output, test_labels[st:end])
        acc_test = accuracy(output, test_labels[st:end],batch=True)
        return loss_test.item(),acc_test.item()

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
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
                #x = F.dropout(x, p=self.dropout, training=self.training)
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
            return x

class ChebNetII(torch.nn.Module):
    def __init__(self,num_features,hidden,num_classes, args):
        super(ChebNetII, self).__init__()
        self.mlp = MLP(num_features,hidden,num_classes,args.num_layers,args.dropout,args.is_bns)
        self.K = args.K
        self.temp = Parameter(torch.Tensor(self.K+1))

        #self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)
        self.mlp.reset_parameters()

    def forward(self, x_lis):
        #coe_tmp=F.relu(self.temp)
        coe_tmp=self.temp
        coe=coe_tmp.clone()
        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)

        Tx_0 = x_lis[0]
        out = coe[0]/2*Tx_0

        for k in range(1,self.K+1):
            Tx_k = x_lis[k]
            out = out+coe[k]*Tx_k
        x=self.mlp(out,input_tensor=True)

        return F.log_softmax(x, dim=1)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default="wiki", help='datasets.')
parser.add_argument('--sub_dataset', type=str, default='')

parser.add_argument('--dev', type=int, default=1, help='device id')
parser.add_argument('--net', type=str, default="ChebNetII", help='device id')
parser.add_argument('--batch_size',type=int, default=10000, help='Batch size')
parser.add_argument('--train_prop', type=float, default=.5,help='training label proportion')
parser.add_argument('--valid_prop', type=float, default=.25,help='validation label proportion')

parser.add_argument('--lr', type=float, default=0.00005, help='learning rate.')       
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')  
parser.add_argument('--early_stopping', type=int, default=300, help='early stopping.')
parser.add_argument('--hidden', type=int, default=2048, help='hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
parser.add_argument('--num_layers', type=int, default=3, help='dropout for neural networks.')
parser.add_argument('--is_bns', type=bool, default=False)

parser.add_argument('--K', type=int, default=10, help='propagation steps.')
parser.add_argument('--pro_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
parser.add_argument('--pro_wd', type=float, default=0.00005, help='learning rate for BernNet propagation layer.')
args = parser.parse_args()
print(args)

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
np.random.seed(0)
batch_size = args.batch_size
device = f'cuda:{args.dev}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dataset = load_nc_dataset(args.dataset)
try:
    cheb_emb = torch.load('./data/cheb_'+args.dataset+'.pt')
except:
	raise RuntimeError('./data/cheb_'+args.dataset+'.pt not found. Need to run python node2vec.py first')

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
labels = dataset.label
if args.dataset == 'wiki':
    split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
else:
    split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)
    split_idx = split_idx_lst[0]
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']

train_data = [mat[train_idx] for mat in cheb_emb]
valid_data = [mat[valid_idx] for mat in cheb_emb]
test_data = [mat[test_idx] for mat in cheb_emb]
train_labels = labels[train_idx].reshape(-1).long().to(device)
valid_labels = labels[valid_idx].reshape(-1).long().to(device)
test_labels = labels[test_idx].reshape(-1).long().to(device)

N = dataset.graph['num_nodes']
num_labels = max(dataset.label.max().item() + 1, dataset.label.shape[1])
num_features = dataset.graph['node_feat'].shape[1]
print(f"num nodes {N} | num classes {num_labels} | num node feats {num_features}")

checkpt_file = './pretrained/'+uuid.uuid4().hex+'.pt'
print(checkpt_file)

model = ChebNetII(num_features,args.hidden,num_labels,args).to(device)
optimizer = torch.optim.Adam([{'params': model.mlp.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.temp, 'weight_decay': args.pro_wd, 'lr': args.pro_lr}])

list_bat_train = create_batch(train_data)
list_bat_val = create_batch(valid_data)
list_bat_test = create_batch(test_data)

bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
valid_num = valid_data[0].shape[0]
test_num = test_data[0].shape[0]

#print('MODEL:', model)
for epoch in range(args.epochs):
    list_loss = []
    list_acc = []
    random.shuffle(list_bat_train)
    for st,end in list_bat_train:
        loss_tra,acc_tra = train(st,end)
        list_loss.append(loss_tra)
        list_acc.append(acc_tra)
    loss_tra = np.round(np.mean(list_loss),4)
    acc_tra = np.round(np.mean(list_acc),4)

    list_loss_val = []
    list_acc_val = []
    for st,end in list_bat_val:
        loss_val,acc_val = validate(st,end)
        list_loss_val.append(loss_val)
        list_acc_val.append(acc_val)

    loss_val = np.round(np.mean(list_loss_val),4)
    acc_val = np.round((np.sum(list_acc_val))/valid_num, 4)

    if epoch%5==0:
        print('train_acc:',acc_tra,'>>>>>>>>>>train_loss:',loss_tra)
        print('val_acc:',acc_val,'>>>>>>>>>>>val_loss:',loss_val)

    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.early_stopping:
        break

list_loss_test = []
list_acc_test = []
model.load_state_dict(torch.load(checkpt_file))
for st,end in list_bat_test:
    loss_test,acc_test = test(st,end)
    list_loss_test.append(loss_test)
    list_acc_test.append(acc_test)
acc_test = (np.sum(list_acc_test))/test_num

#print(name)
print('Load {}th epoch'.format(best_epoch))
print(f"Valdiation accuracy: {np.round(acc*100,2)}, Test accuracy: {np.round(acc_test*100,2)}")

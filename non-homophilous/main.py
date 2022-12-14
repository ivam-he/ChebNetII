import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import seaborn as sns
from torch_geometric.utils import to_undirected
from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import evaluate, eval_acc, eval_rocauc, load_fixed_splits
from utils import set_seed
from models import ChebNetII,ChebNetII_V

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser.add_argument('--seed',type=int, default=42)
parser.add_argument('--dev',type=int, default=0)
parser.add_argument('--dataset', type=str, default='fb100')
parser.add_argument('--sub_dataset', type=str, default='')

parser.add_argument('--K', type=int, default=10)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2, help='number of layers for MLP')
parser.add_argument('--net', type=str, default='ChebNetII')

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--runs', type=int, default=5,help='number of distinct runs')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dprate', type=float, default=0.5)
parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
parser.add_argument('--prop_wd', type=float, default=0.0005, help='weight decay for propagation layer.')
parser.add_argument('--is_bns', type=bool, default=False)
args = parser.parse_args()
print(args)
print("---------------------------------------------")
set_seed(args.seed)

device = f'cuda:{args.dev}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

#load fixed dataset split
split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)
n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)
print(f"num nodes {n} | num classes {c} | num node feats {d}")

if args.dataset =="twitch-gamer":
    model = ChebNetII_V(d, c, args).to(device)
else:
    model =ChebNetII(d, c, args).to(device)
if args.dataset == 'genius':
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc
logger = Logger(args.runs, args)
model.train()
### Training loop ###
results = []
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    if args.dataset == "twitch-gamer":
        optimizer = torch.optim.AdamW([{ 'params': model.mlp.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.mlp1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': args.prop_wd, 'lr': args.prop_lr}])
    else:
        optimizer = torch.optim.AdamW([{ 'params': model.mlp.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': args.prop_wd, 'lr': args.prop_lr}])

    best_val_acc = best_test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset)
        if args.dataset =='genius':
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])     
        loss.backward()
        optimizer.step()
        train_acc, val_acc, test_acc = evaluate(model, dataset, split_idx, eval_func, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        if epoch % 50 == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * val_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%')
    print(f'run:{run:02d}',f'Test_acc:{100*best_test_acc:.2f}%')
    results.append([best_test_acc,best_val_acc])

test_acc_mean, val_acc_mean= np.mean(results, axis=0) * 100
test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
values=np.asarray(results)[:,0]
uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))
print(f'Dataset {args.dataset}, in {args.runs} repeated experiment:')
print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
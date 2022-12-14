import random
import argparse
import torch
import uuid
import pickle
import numpy as np
import torch.optim as optim
import scipy.sparse as sp
import torch.nn.functional as F
from model import ChebNetII
from utils import accuracy, set_seed

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--data', type=str, default="papers100m", help='datasets.')

parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--net', type=str, default="ChebNetII", help='device id')
parser.add_argument('--batch_size',type=int, default=10000, help='Batch size')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')       
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')  
parser.add_argument('--early_stopping', type=int, default=300, help='early stopping.')
parser.add_argument('--hidden', type=int, default=2048, help='hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

parser.add_argument('--K', type=int, default=10, help='propagation steps.')
parser.add_argument('--pro_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
parser.add_argument('--pro_wd', type=float, default=0.00005, help='learning rate for BernNet propagation layer.')
args = parser.parse_args()
print(args)
set_seed(args.seed)

batch_size = args.batch_size
name = args.data
data_path = './data/'
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)

with open(data_path+"training_"+name+".pickle","rb") as fopen:
    train_data = pickle.load(fopen)

with open(data_path+"validation_"+name+".pickle","rb") as fopen:
    valid_data = pickle.load(fopen)

with open(data_path+"test_"+name+".pickle","rb") as fopen:
    test_data = pickle.load(fopen)

with open(data_path+"labels_"+name+".pickle","rb") as fopen:
    labels = pickle.load(fopen)

train_data = [mat.to(device) for mat in train_data[:args.K+1]]
valid_data = [mat.to(device) for mat in valid_data[:args.K+1]]
test_data = [mat.to(device) for mat in test_data[:args.K+1]]
train_labels = labels[0].reshape(-1).long().to(device)
valid_labels = labels[1].reshape(-1).long().to(device)
test_labels = labels[2].reshape(-1).long().to(device)


num_features = train_data[0].shape[1]
num_labels = int(train_labels.max()) + 1
print("Number of labels for "+name, num_labels)

checkpt_file = './pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

model = ChebNetII(num_features,args.hidden,num_labels,args).to(device)
optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin3.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.temp, 'weight_decay': args.pro_wd, 'lr': args.pro_lr}])

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
    output = model(train_data,st,end)
    acc_train = accuracy(output, train_labels[st:end])
    loss_train = F.nll_loss(output, train_labels[st:end])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()

def validate(st,end):
    model.eval()
    with torch.no_grad():
        output = model(valid_data,st,end)
        loss_val = F.nll_loss(output, valid_labels[st:end])
        acc_val = accuracy(output, valid_labels[st:end],batch=True)
        return loss_val.item(),acc_val.item()

def test(st,end):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        output = model(test_data,st,end)
        loss_test = F.nll_loss(output, test_labels[st:end])
        acc_test = accuracy(output, test_labels[st:end],batch=True)
        return loss_test.item(),acc_test.item()

list_bat_train = create_batch(train_data)
list_bat_val = create_batch(valid_data)
list_bat_test = create_batch(test_data)

bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
valid_num = valid_data[0].shape[0]
test_num = test_data[0].shape[0]
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
    acc_val = np.round((np.sum(list_acc_val))/valid_num,4)

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

print(name)
print('Load {}th epoch'.format(best_epoch))
print(f"Valdiation accuracy: {np.round(acc*100,2)}, Test accuracy: {np.round(acc_test*100,2)}")

    







from igraph.drawing import graph
import numpy as np
import time
import networkx as nx
import torch
from torch._C import device
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import heapq as hp

from torch_geometric.loader.data_list_loader import DataListLoader

from data_reader2 import DataReader
from models import GNN

import random
import os
import os.path as osp
import json
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm

# =========== fix random seed ============
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def check_bad_model(edge_standard, ds_name, method, snapshot, pooling, lr):
    if edge_standard == 'attr' or edge_standard == 'vattr':
        return
    if edge_standard != 'degeneracy' and int(snapshot) == 1:
        print('only degeneracy to make it faster!')
        exit(0)
    if ds_name == 'DD' and lr == 0.01 and int(snapshot) > 1 and pooling == 'mean' \
        and method == 'intact' and edge_standard != 'curvature':
        print('bad model!')
        exit(0)


# Experiment parameters
'''
----------------------------
Dataset  |   batchnorm_dim
----------------------------
MUTAG    |     28
PTC_MR   |     64
BZR      |     57
COX2     |     56
COX2_MD  |     36
BZR-MD   |     33
PROTEINS |    620
D&D      |   5748
'''

batch_norm_dim = {
    'MUTAG' : 28,
    'PTC_MR': 64,
    'BZR': 57,
    'COX2': 56,
    'COX2_MD': 36,
    'BZR_MD': 33,
    'PROTEINS': 620,
    'DD': 5748,
}

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', help='Select CPU/CUDA for training.')
parser.add_argument('--dataset', default='MUTAG', help='Dataset name.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')

#random splits: 0.008 -> MUTAG and RDTM5K | 0.007 -> PTC_MR, BZR, COX2, PROTEINS, IMDB-B and D&D
#stratified splits: 0.009 for all datasets
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.') 

parser.add_argument('--wdecay', type=float, default=9e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--hidden-dim', type=int, default=64, help='Number of hidden units.')
# FIXME: ignore batch norm dim
parser.add_argument('--batchnorm_dim', type=int, default=0, help='Batchnormalization dimension for GraphSN layer.')

parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for MLP layers in GraphSN.')
parser.add_argument('--n_folds', type=int, default=10, help='Number of folds in cross validation.')
parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
parser.add_argument('--log_interval', type=int, default=10 , help='Log interval for visualizing outputs.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')


parser.add_argument('--n_layers', type=int, default=2, help='Number of MLP layers for GraphSN.')
# parser.add_argument('--run', default=0,type=int,required=True,help='[0, 9]')
parser.add_argument('--pooling',default='sum',required=True,help='mean / sum')
parser.add_argument('--edge-standard',default='curvature',required=True,help='cn / degeneracy / curvature ..')
parser.add_argument('--method',default='intact',required=True,help='intact / partial')
parser.add_argument('--snapshot',default='001',required=True,help='number of snapshot')
parser.add_argument('--file',required=True)

args = parser.parse_args()

num_threads = 1

print('#thread {}'.format(num_threads))
torch.set_num_threads(num_threads)

# args.batchnorm_dim = batch_norm_dim[args.dataset]
edge_standard = args.edge_standard
method = args.method
snapshot = args.snapshot
dataset = args.dataset

path = osp.dirname(osp.abspath(__file__))
res_json_filename = args.file
graph_name = f'{dataset}-{method}-{int(snapshot):0>3d}'
method_name = f'{args.pooling}-{args.n_layers}-L-{args.dropout}'

if osp.exists(res_json_filename):
    with open(res_json_filename, 'r') as f:
        res_json = json.load(f)
else:
    res_json = {}

if graph_name not in res_json:
    res_json[graph_name] = {}
if method_name not in res_json[graph_name]:
    res_json[graph_name][method_name] = {}
    res_json[graph_name][method_name]['acc'] = []
    res_json[graph_name][method_name]['time'] = []

if len(res_json[graph_name][method_name]['acc']) == 10:
    print('exist! continue!')
    exit(0)
else:
    res_json[graph_name][method_name]['acc'] = []
    res_json[graph_name][method_name]['time'] = []

check_bad_model(edge_standard,args.dataset,method,snapshot,args.pooling,args.lr)

print('Loading data ... ')


datareader = DataReader(dataset=args.dataset,
                        edge_standard=args.edge_standard,
                        method=args.method,
                        snapshot=args.snapshot)


n_graphs = datareader.data['n_graphs']
data_list = []
for itr in tqdm(range(n_graphs), desc='calc structral coef'):

    row, col = datareader.data['adj_list'][itr]
    w = datareader.data['edge_weight'][itr]
    num_of_nodes = len(datareader.data['features'][itr])
    adj = torch.zeros(num_of_nodes, num_of_nodes)
    new_adj = torch.zeros(num_of_nodes, num_of_nodes)

    assert row.shape[0] == w.shape[0]
    for i in np.arange(row.shape[0]):
        adj[row[i]][col[i]]=1.0
        new_adj[row[i]][col[i]]=w[i]

    A_array = adj.detach().numpy()
    weight = torch.FloatTensor(new_adj)
    weight = weight / weight.sum(1, keepdim=True)
    weight = weight + torch.FloatTensor(A_array)

    weight = weight.detach().numpy()
    weight = np.nan_to_num(weight, nan=0)
    weight = torch.FloatTensor(weight)
    coeff = weight.sum(1, keepdim=True)
    coeff = (coeff.T)[0]

    weight = weight.detach().numpy()

    # print(datareader.data['features_onehot'][itr].dtype)
    data = Data(x=torch.from_numpy(datareader.data['features_onehot'][itr]), 
                edge_index=torch.from_numpy(datareader.data['adj_list'][itr]), 
                y=torch.tensor(datareader.data['targets'][itr]))
    data.norm_edge_weight = torch.FloatTensor(weight[np.nonzero(weight)])
    data.norm_self_loop = torch.FloatTensor(coeff)
    data.num_nodes = datareader.data['num_nodes'][itr]
    # data.num_features = datareader.data['features_dim']
    # data.num_classes = datareader.data['n_classes']

    data_list.append(data)

num_features = datareader.data['features_dim']
num_classes = datareader.data['n_classes']

all_test_acc = []
for fold_id in range(10):
    loaders = []
    for split in ['train', 'valid', 'test']:
        gdata = []
        for x in datareader.data['splits'][fold_id][split]:
            gdata.append(data_list[x])
        loader = DataLoader(gdata,batch_size=args.batch_size,shuffle=('train' in split))
        loaders.append(loader)

    model = GNN(num_tasks=num_classes,
                in_dim=num_features,
                num_layer=args.n_layers,
                emb_dim=args.hidden_dim,
                gnn_type='graphSN',
                drop_ratio=args.dropout,
                graph_pooling=args.pooling).to(args.device)

    # print('\nInitialize model')
    # print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wdecay,
                betas=(0.5, 0.999))

    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)

    def train(train_loader):
        model.train()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            # print(output, data)
            loss = loss_fn(output, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(output)
            n_samples += len(output)
        scheduler.step()
        return train_loss / n_samples
        
    @torch.no_grad()
    def test(loader):
        model.eval()
        total_loss, correct, n_samples = 0, 0, 0
        for batch_idx, data in enumerate(loader):
            data = data.to(args.device)
            output = model(data)
            loss = loss_fn(output, data.y)
            total_loss += loss.item()
            n_samples += len(output)
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            correct += pred.eq(data.y.detach().cpu().view_as(pred)).sum().item()

        total_loss /= n_samples

        acc = 100. * correct / n_samples
        return total_loss, acc

    loss_fn = F.cross_entropy
    best_valid_loss = 1e9
    patience = 0
    best_acc = 0
    time_per_epoch = []
    for epoch in range(args.epochs):
        tic = time.process_time()
        train_loss = train(loaders[0])
        valid_loss, valid_acc = test(loaders[1])
        test_loss, test_acc = test(loaders[2])

        time_cost = time.process_time() - tic
        time_per_epoch.append(time_cost)
        patience += 1

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_acc = test_acc
            patience = 0
            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f} '
                    f'Test Acc: {test_acc:.4f}, Time: {time_cost:.2f}(s)[*]')
        
        if patience > 50:
            print(f'Best Acc: {best_acc:.4f}\n')
            print('early stopping!')
            break
        
        if epoch % args.log_interval == 0:
            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f} '
                    f'Test Acc: {test_acc:.4f}, Time: {time_cost:.2f}(s)')

    print(f'{args.dataset}-{edge_standard}-{method}-{int(snapshot):0>3d}  Fold {fold_id}: Best Acc: {best_acc:.4f}\n')
    all_test_acc.append(best_acc)
    res_json[graph_name][method_name]['acc'].append(best_acc / 100)
    res_json[graph_name][method_name]['time'].append(np.array(time_per_epoch).mean())
    with open(res_json_filename, 'w') as f:
        res = json.dumps(res_json, indent=4)
        f.write(res)

acc = np.array(all_test_acc).mean()
std = np.array(all_test_acc).std()
print(f'10 folds answer: {acc:2f} +- {std:.2f}')

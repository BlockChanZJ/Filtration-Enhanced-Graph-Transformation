import glob
import json
from math import e
import os
import os.path as osp
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, pool
import argparse
import igraph as ig
import time

from torch_geometric.datasets import TUDataset
from dataloader import *

class GIN(torch.nn.Module):

    def __init__(self, d_in, d_hidden, d_out, n_layers, pooling, dropout):
        super(GIN, self).__init__()

        self.convs = []
        self.linears = []
        self.n_layers = n_layers

        if pooling == 'sum':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool

        self.dropout = dropout

        assert n_layers >= 2

        for layer in range(n_layers):
            if layer == 0:
                mlp = Sequential(Linear(d_in, d_hidden), BatchNorm1d(d_hidden), ReLU(),
                                 Linear(d_hidden, d_hidden), ReLU())
            elif layer == n_layers - 1:
                mlp = Sequential(Linear(d_hidden, d_hidden), BatchNorm1d(d_hidden), ReLU(),
                                 Linear(d_hidden, d_hidden), ReLU())
            else:
                mlp = Sequential(Linear(d_hidden, d_hidden), BatchNorm1d(d_hidden), ReLU(),
                                 Linear(d_hidden, d_hidden), ReLU())
            self.convs.append(GINConv(mlp))

        self.convs = torch.nn.ModuleList(self.convs)
        self.linear1 = Linear(d_hidden, d_hidden)
        self.linear2 = Linear(d_hidden, d_out)

    def forward(self, x, edge_index, batch):

        out = 0

        for layer in range(self.n_layers):
            x = self.convs[layer](x, edge_index)
            out += F.dropout(self.pooling(x, batch),
                             p=self.dropout, training=self.training)
        out = self.linear2(out)
        return F.log_softmax(out, dim=-1)


def train(model, train_loader):
    model.train()

    total_loss = 0
    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(train_loader)


@torch.no_grad()
def valid(model, valid_loader):
    model.eval()

    total_loss = 0
    for data in valid_loader:
        data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(output, data.y)
        total_loss += loss
    return total_loss / len(valid_loader)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int((out.argmax(-1) == data.y).sum())
    return total_correct / len(loader)


if __name__ == '__main__':

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

    # =========== args ============
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='MUTAG',required=True,help='dataset')
    parser.add_argument('--layer',type=int,default=3,help='number of layers')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--method',required=True,help='intact or partial')
    parser.add_argument('--snapshot',required=True,help='001 002 ...')
    parser.add_argument('--epochs',default=500,type=int,help='number of epochs')
    parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
    parser.add_argument('--pooling',default='mean',help='sum or mean')
    parser.add_argument('--run',default=0,type=int,help='[0, 9] (used for 10folds)')
    parser.add_argument('--layer-feature',action='store_true',help='whether to use layer feature')
    parser.add_argument('--hidden-dim',default=64,type=int)
    parser.add_argument('--dropout',default=0,type=float)
    parser.add_argument('--file',required=True)
    parser.add_argument('--edge-standard',required=True,help='curvature')
    args = parser.parse_args()

    epochs = args.epochs
    ds_name = args.dataset
    method = args.method
    snapshot = args.snapshot
    batch_size = args.batch_size
    n_layers = args.layer
    lr = args.lr
    current_run = args.run
    pooling = args.pooling
    edge_standard = args.edge_standard
    use_layer_feature = args.layer_feature
    hidden_dim = args.hidden_dim
    device = 'cpu'
    dropout = args.dropout
    file = args.file
    print(args)

    num_threads = 1
    print('#thread {}'.format(num_threads))
    torch.set_num_threads(num_threads)
    
    # =========== load dataset ============
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    filtrated_path = osp.join(path, args.dataset, 'filtrated')
    ds_file = osp.join(filtrated_path,'{}-{}-{:0>3}.pt'.format(args.edge_standard,args.method,int(args.snapshot)))
    if not osp.exists(ds_file):
        print('please build filtration graph first!')
        exit(1)

    graphs = torch.load(ds_file)
    dataset = [Data(graph) for graph in graphs]
    del graphs

    infos = get_info(ds_name, dataset, snapshot)
    d_in = infos['d_in']
    n_classes = infos['n_classes']
    n_labels = infos['n_labels']
    one_hot = infos['one_hot']
    link_edge = infos['link_edge']

    print('# link edge: {:.3f}'.format(link_edge))


    if osp.exists(file):
        with open(file, 'r') as f:
            res_dict = json.load(f)
    else:
        res_dict = {}

    json_name = '{}-{}-{:0>3}'.format(ds_name, method, (int)(snapshot))
    model_name = '{}-{}-{}-{}'.format(pooling, n_layers,
                                      'L' if use_layer_feature else 'NL', dropout)




    if json_name not in res_dict:
        res_dict[json_name] = {}
    if model_name not in res_dict[json_name]:
        res_dict[json_name][model_name] = {}
        res_dict[json_name][model_name]['acc'] = []
        res_dict[json_name][model_name]['time'] = []

    if len(res_dict[json_name][model_name]['acc']) > (int)(current_run):
        print('continue!')
        exit(0)

    train_dataset, valid_dataset, test_dataset = get_split_dataset(
        ds_name, dataset, current_run)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, n_labels=n_labels, n_snapshots=int(
        snapshot), use_layer=use_layer_feature, one_hot=one_hot)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, n_labels=n_labels, n_snapshots=int(
        snapshot), use_layer=use_layer_feature, one_hot=one_hot)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, n_labels=n_labels, n_snapshots=int(
        snapshot), use_layer=use_layer_feature, one_hot=one_hot)



    if use_layer_feature:
        model = GIN(d_in=d_in, d_hidden=hidden_dim,
                    d_out=n_classes, n_layers=n_layers, pooling=pooling, dropout=dropout).to(device)
    else:
        model = GIN(d_in=d_in, d_hidden=hidden_dim, d_out=n_classes,
                    n_layers=n_layers, pooling=pooling, dropout=dropout).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=10,
                                                           factor=0.5,
                                                           verbose=True)

    best_valid_loss = 1e9
    best_test_acc = 0
    patience = 0

    time_per_epoch_list = []
    tic = time.process_time()
    for epoch in range(1, epochs + 1):
        time_per_epoch = time.process_time()
        model.train()

        train_loss = train(model, train_loader)
        valid_loss = valid(model, valid_loader)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        scheduler.step(train_loss)
        time_per_epoch_list.append(time.process_time() - time_per_epoch)
        if valid_loss < best_valid_loss:
            patience = 0
            best_valid_loss = valid_loss
            best_test_acc = test_acc
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Valid Loss:{valid_loss:.4f}, Train Acc: {train_acc:.4f} '
                  f'Test Acc: {test_acc:.4f}, Time: {time.process_time() - tic:.2f}(s)[*]')
        if epoch % 10 == 1:
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Valid Loss:{valid_loss:.4f}, Train Acc: {train_acc:.4f} '
                  f'Test Acc: {test_acc:.4f}, Time: {time.process_time() - tic:.2f}(s)')
            tic = time.process_time()
        patience += 1
        if patience > 50:
            break
    print(f'Best Acc: {best_test_acc:.4f}\n')
    res_dict[json_name][model_name]['acc'].append(best_test_acc)
    res_dict[json_name][model_name]['time'].append(
        np.array(time_per_epoch_list).mean())

    res = json.dumps(res_dict, indent=4)
    with open(file, 'w') as f:
        f.write(res)

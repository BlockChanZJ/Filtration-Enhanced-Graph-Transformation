from os import link
from typing import List

import numpy as np
import torch
from torch.functional import split
import torch.nn.functional as F
import igraph as ig

DIRECT_LINK_EDGE = False

class Data():
    def __init__(self, graph: ig.Graph):
        if 'feat' in graph.vs.attributes():
            self.x = graph.vs['feat']
        else:
            self.x = [0 for _ in range(graph.vcount())]
        assert 'layer' in graph.vs.attributes()
        self.layer = graph.vs['layer']
        self.edges = []
        self.link_edge = 0
        for e in graph.es:
            u, v = e.tuple
            lu, lv = self.layer[u], self.layer[v]
            if lu != lv:
                self.link_edge += 1
            if DIRECT_LINK_EDGE:
                if lu <= lv:
                    self.edges.append((u, v))
                if lv <= lu:
                    self.edges.append((v, u))
            else:
                self.edges.append((u, v))
                self.edges.append((v, u))
        self.y = graph['label']


class BatchData():
    def __init__(self, data_list: List[Data], n_labels, n_snapshots, use_layer, one_hot=True):
        x = []
        y = []
        edge_index = [[], []]
        layer = []
        batch = []
        base_idx = 0
        for i, data in enumerate(data_list):
            x.extend(data.x)
            y.append(data.y)
            layer.extend(data.layer)
            edge_index[0].extend([e[0] + base_idx for e in data.edges])
            edge_index[1].extend([e[1] + base_idx for e in data.edges])
            base_idx += len(data.x)
            batch.extend([i for _ in range(len(data.x))])
        if one_hot:
            self.x = torch.tensor(F.one_hot(torch.tensor(
                x), num_classes=n_labels), dtype=torch.float)
            if use_layer:
                layer = torch.tensor(F.one_hot(torch.tensor(
                    layer), num_classes=n_snapshots), dtype=torch.float)
                layer = self.label_smooth(layer, 0.2)
                self.x = torch.cat([self.x, layer], dim=1)
        else:
            if not isinstance(x[0], List):
                self.x = torch.tensor(x, dtype=torch.long).reshape(len(x), 1)
                if use_layer:
                    layer = torch.tensor(layer, dtype=torch.long).reshape(len(layer), 1)
                    self.x = torch.cat([self.x, layer], dim=1)
            else:
                self.x = torch.tensor(x, dtype=torch.float)
                if use_layer:
                    layer = torch.tensor(layer, dtype=torch.float).reshape(len(layer), 1)
                    self.x = torch.cat([self.x, layer], dim=1)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.batch = torch.tensor(batch, dtype=torch.long)

    def label_smooth(self, x, epsilon=0.1):
        return x.mul(1-epsilon) + x.div(x.shape[1])

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return 'x: {}\n'.format(self.x.shape) + \
               'edge_index: {}\n'.format(self.edge_index.shape) + \
               'y: {}\n'.format(len(self.y))


class DataSampler(object):
    def __init__(self, batch_size, data_size):
        self.batch_size = batch_size
        self.data_size = data_size
        self.indices = list(range(0, data_size))
        self.index = 0

    def __iter__(self):
        indices = self.indices
        batch = []
        for i in indices:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return self.data_size


class DataLoader():
    def __init__(self, data: List[Data], n_labels, n_snapshots, batch_size=1, seed=42, use_layer=True, one_hot=True):
        self.batch_size = batch_size
        self.seed = seed
        self.sample_iter = DataSampler(
            batch_size=batch_size, data_size=len(data))
        self.data = data
        self.use_layer = use_layer
        self.n_labels = n_labels
        self.n_snapshots = n_snapshots
        self.one_hot = one_hot

    def __iter__(self):
        for indices in self.sample_iter:
            batch_data = BatchData([self.data[i] for i in indices], n_labels=self.n_labels,
                                   n_snapshots=self.n_snapshots, use_layer=self.use_layer, one_hot=self.one_hot)
            yield batch_data

    def __len__(self):
        return len(self.data)


def get_split_dataset(ds_name, dataset, run):
    chemical_ds = ['DD', 'ENZYMES', 'NCI1', 'PROTEINS']
    social_ds = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
                 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    import json
    import os.path as osp
    import random
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data_splits')
    if ds_name in chemical_ds or ds_name in social_ds:
        if ds_name in chemical_ds:
            with open(osp.join(path, 'chemical', ds_name+'_splits.json'), 'r') as f:
                splits = json.load(f)
        else:
            with open(osp.join(path, 'social', ds_name+'_splits.json'), 'r') as f:
                splits = json.load(f)
        train_idx = splits[run]['model_selection'][0]['train']
        valid_idx = splits[run]['model_selection'][0]['validation']
        test_idx = splits[run]['test']
        print('train: {}, valid: {}, test: {}'.format(len(train_idx), len(valid_idx), len(test_idx)))
    else:
        # # FIXME: run = 0
        # run = 0
        n_graphs = len(dataset)
        splits = list(range(n_graphs))
        random.shuffle(splits)
        splits_l = splits[:n_graphs // 10 * run]
        splits = splits[n_graphs // 10 * run:]
        splits.extend(splits_l)
        train_idx = splits[n_graphs // 10 * 2:]
        valid_idx = splits[n_graphs // 10: n_graphs // 10 * 2]
        test_idx = splits[:n_graphs // 10]
        print('train: {}, valid: {}, test: {}'.format(len(train_idx), len(valid_idx), len(test_idx)))
    train_dataset, valid_dataset, test_dataset = [], [], []
    for x in train_idx:
        train_dataset.append(dataset[x])
    for x in valid_idx:
        valid_dataset.append(dataset[x])
    for x in test_idx:
        test_dataset.append(dataset[x])
    return train_dataset, valid_dataset, test_dataset


def get_info(ds_name, dataset, snapshot):
    n_max_node = 0
    link_edge = 0
    if ds_name == 'ENZYMES' or ds_name == 'BZR' or ds_name == 'COX2' or ds_name == 'DHFR':
        n_labels, n_classes = len(dataset[0].x[0]), set()
        for data in dataset:
            n_classes.add(data.y)
            n_max_node = max (n_max_node, len(data.x))
            link_edge += data.link_edge
        n_classes = len(n_classes)
        one_hot = False
        d_in = n_labels + 1
    else:
        n_labels, n_classes = 0, set()
        for data in dataset:
            link_edge += data.link_edge
            n_labels = max(n_labels, max(data.x) + 1)
            n_classes.add(data.y)
            n_max_node = max (n_max_node, len(data.x))
        n_classes = len(n_classes)
        one_hot = True
        d_in = n_labels + (int)(snapshot)

    ans = {
        'd_in' : d_in,
        'n_labels' : n_labels,
        'n_classes' : n_classes,
        'one_hot' : one_hot,
        'n_max_node': n_max_node,
        'link_edge': link_edge / len(dataset)
    } 
    return ans
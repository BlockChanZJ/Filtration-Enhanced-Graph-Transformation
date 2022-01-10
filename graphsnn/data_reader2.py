from typing import List
import numpy as np
import os
import math
from os.path import join as pjoin
import torch
import os.path as osp
import igraph as ig

class DataReader():
    '''
    Class to read the txt files containing all data of the dataset
    '''
    def __init__(self,
                 dataset,
                 edge_standard,
                 method,
                 snapshot):
        path = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'data', dataset, 'filtrated')
        filename = osp.join(path, f'{edge_standard}-{method}-{int(snapshot):0>3d}.pt')
        graphs = torch.load(filename)
        
        self.reweight_edges_with_overlap(graphs)
        data = {}
        data['adj_list'], data['edge_weight'] = self.parse_adj(graphs)
        data['features'] = self.parse_node_features(graphs)
        data['layers'] = self.parse_node_layers(graphs)
        data['targets'] = np.array([graph['label'] for graph in graphs])
        data['n_graphs'] = len(graphs)

        self.n_graphs = len(graphs)
           
        features, layers, n_edges, degrees = [], [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            n = len(adj[0])  # total sum of edges
            assert n % 2 == 0, n
            n_edges.append( int(n / 2) )  # undirected edges, so need to divide by 2
            # if not np.allclose(adj, adj.T):
            #     print(sample_id, 'not symmetric')
            degrees.extend(list(graphs[sample_id].degree()))
            features.append(np.array(data['features'][sample_id]))
            layers.append(np.array(data['layers'][sample_id]))
                        
        # Create features over graphs as one-hot vectors for each node or native attributes
        features_all = np.concatenate(features)
        layers_all = np.concatenate(layers)
        features_min = features_all.min()
        layers_min = layers_all.min()
        
        if dataset == 'BZR' or dataset == 'COX2' or dataset == 'DHFR' or dataset == 'ENZYMES':
            features_dim = features_all.shape[1]
            use_one_hot = False
        else:
            features_dim = int(features_all.max() - features_min + 1)  # number of possible values
            use_one_hot = True
        layers_dim = int(layers_all.max() - layers_min + 1)
        
        features_onehot = []
        assert len(features) == len(layers)
        # print(features)
        for i, t in enumerate(zip(features, layers)):
            x, y = t[0], t[1]
            assert len(x) == len(y)
            if use_one_hot:
                feature_onehot = np.zeros((len(x), features_dim),dtype=np.float32)
                layer_onehot = np.zeros((len(y), layers_dim),dtype=np.float32)
                for node, value in enumerate(x):
                    feature_onehot[node, value - features_min] = 1
                for node, value in enumerate(y):
                    layer_onehot[node, value - layers_min] = 1
                
                features_onehot.append(np.concatenate([feature_onehot, layer_onehot], axis=1))
            else:
                layer_onehot = np.zeros((len(y), layers_dim),dtype=np.float32)
                for node, value in enumerate(y):
                    layer_onehot[node, value - layers_min] = 1
                # print(x)
                print(x.shape, layer_onehot.shape)
                features_onehot.append(np.concatenate([x, layer_onehot],axis=1,dtype=np.float32))

        features_dim += layers_dim
        print(f'one hot feature: {use_one_hot}')
        assert features_dim == features_onehot[0].shape[1]
            
        shapes = [graph.vcount() for graph in graphs]
        labels = data['targets']        # graph class labels
        labels -= np.min(labels)        # to start from 0

        classes = np.unique(labels)
        n_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), 
                                                              np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), 
                                                              np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), 
                                                                  np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        if use_one_hot:
            for u in np.unique(features_all):
                print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        for u in np.unique(layers_all):
            print('layer {}, count {}/{}'.format(u, np.count_nonzero(layers_all == u), len(layers_all)))
        

        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        # splits from text
        train_ids, valid_ids, test_ids = self.split_ids(dataset)
        
        # Create train sets
        splits = []
        for fold in range(10):
            splits.append({'train': train_ids[fold],
                           'valid': valid_ids[fold],
                           'test': test_ids[fold]})
            
        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits 
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes
        data['num_nodes'] = shapes
        
        self.data = data
    
    def parse_adj(self,graphs: List[ig.Graph]):
        adj_dict, weight_dict = [], []
        for graph_id, graph in enumerate(graphs):
            tuples = [e.tuple for e in graph.es]
            weights = graph.es['weight']
            edge_tuple = sorted(list(set([(x[0], x[1], y) if x[0] <= x[1] else (x[1], x[0], y) for x, y in zip(tuples, weights)])))
            edges = [[], []]
            weights = []
            for u, v, w in edge_tuple:
                if u == v: continue
                edges[0].append(u)
                edges[1].append(v)
                weights.append(w)
                edges[0].append(v)
                edges[1].append(u)
                weights.append(w)
            weight_dict.append(np.array(weights))
            adj_dict.append(np.array(edges))
        return adj_dict, weight_dict


    def parse_node_features(self,graphs: List[ig.Graph]):
        node_features = []
        for graph_id, graph in enumerate(graphs):
            node_features.append(graph.vs['feat'])
        return node_features
    
    def parse_node_layers(self,graphs:List[ig.Graph]):
        node_layers = []
        for graph_id, graph in enumerate(graphs):
            node_layers.append(graph.vs['layer'])
        return node_layers
    
    def split_ids(self, dataset):
        train_ids, valid_ids, test_ids = [], [], []
        for run in range(10):
            train_idx, valid_idx, test_idx = self.get_split_dataset(dataset, run)
            train_ids.append(np.array(train_idx))
            valid_ids.append(np.array(valid_idx))
            test_ids.append(np.array(test_idx))
        return train_ids, valid_ids, test_ids

    
    def get_split_dataset(self, ds_name, run):
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
        else:
            n_graphs = self.n_graphs
            splits = list(range(n_graphs))
            random.shuffle(splits)
            splits_l = splits[:n_graphs // 10 * run]
            splits = splits[n_graphs // 10 * run:]
            splits.extend(splits_l)
            train_idx = splits[n_graphs // 10 * 2:]
            valid_idx = splits[n_graphs // 10: n_graphs // 10 * 2]
            test_idx = splits[:n_graphs // 10]
        return train_idx, valid_idx, test_idx
    

    def reweight_edges_with_overlap(self, graphs: List[ig.Graph]):
        import time
        tic = time.time()
        for graph in graphs:
            neighbors = [set(graph.neighbors(vertex=u)) for u in graph.vs]
            for i in range(len(neighbors)):
                neighbors[i].add(i)
            edge_weights = []
            for e in graph.es:
                u, v = e.tuple
                common_neighbors = neighbors[u].intersection(neighbors[v])
                common_edges = 0
                for x in common_neighbors:
                    for y in common_neighbors:
                        if y in neighbors[x]:
                            common_edges += 1
                cns = len(common_neighbors)
                assert cns > 1
                edge_weights.append(common_edges * (cns ** 2) / (cns * (cns - 1)))
            graph.es['weight'] = edge_weights

        print('calc overlap time: {:.2f}(s)'.format(time.time() - tic))
    

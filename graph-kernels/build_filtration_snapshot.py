import copy
import glob
import time
from operator import add

import igraph as ig
import networkx as nx
import numpy as np
import sklearn.preprocessing
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from numpy.lib.function_base import append
from pandas.core import base
from tqdm import tqdm
from typing import List
from dataset import load_tu_graphs
from utils import *
from filtration import *
import argparse
import os.path as osp
import os
from config import *
import torch


USE_LINK_EDGE = True
FORCE_MODE = False
LINK_EDGE_WEIGHT = 0
link_edge = 0

def build_filtrated_snapshots(graph: ig.Graph, method, snapshot_list: List):
    if len(snapshot_list) == 2:
        graph.vs['feat'] = graph.vs['weight']
        graph.vs['layer'] = [0 for _ in range(graph.vcount())]
        return graph

    assert 'weight' in graph.es.attributes()
    assert 'weight' in graph.vs.attributes()

    res_snapshots = []
    n_snapshots = len(snapshot_list) - 1
    
    edge_weights = np.array(graph.es['weight'])
    edge_indices = np.argsort(edge_weights, kind='stable')

    for snap in range(n_snapshots):
        filter_value_l = snapshot_list[snap]
        filter_value_r = snapshot_list[snap + 1]
        idx_l, idx_r = find_interval(edge_weights[edge_indices], filter_value_l, filter_value_r)
        idx = range(idx_l, idx_r)

        if method == 'intact':
            idx = range(0, idx_r)

        vid_map, vid_cnt = {}, 0

        for i, (edge_index, edge_weight) in enumerate(zip(edge_indices[idx], edge_weights[edge_indices][idx])):
            u, v = graph.es[edge_index].tuple
            if u not in vid_map:
                vid_map[u] = vid_cnt
                vid_cnt += 1
            if v not in vid_map:
                vid_map[v] = vid_cnt
                vid_cnt += 1
        
        node_feat_list = []
        for k, v in vid_map.items():
            node_feat_list.append(graph.vs[k]['weight'])

        
        if vid_cnt < 2:
            # 1. avoid grakel error
            # 2. \delta similarity between empty snapshots 
            # assert vid_cnt == 1
            graph_snapshot = ig.Graph(n=2)
            graph_snapshot.vs['feat'] = [-1, -1]
            # graph_snapshot.vs['feat'] = [random.randint(-1e9, -1), random.randint(-1e9, -1)]
            graph_snapshot['label'] = graph['label']
            graph_snapshot.vs['layer'] = [0, 0]
            graph_snapshot.add_edge(0, 1)
            res_snapshots.append(graph_snapshot)
            continue
        else:
            graph_snapshot = ig.Graph(n=vid_cnt)
            graph_snapshot.vs['feat'] = node_feat_list
            graph_snapshot['label'] = graph['label']
            graph_snapshot.vs['layer'] = [0 for _ in range(vid_cnt)]

        for i, (edge_index, edge_weight) in enumerate(zip(edge_indices[idx], edge_weights[edge_indices][idx])):
            u, v = graph.es[edge_index].tuple
            graph_snapshot.add_edge(vid_map[u], vid_map[v], weight=edge_weight)
        
        res_snapshots.append(graph_snapshot)

    return res_snapshots





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='MUTAG',
        required=True,
        help='dataset'
    )
    
    parser.add_argument(
        '--edge-standard',
        default='label',
        help='label, attr, curvature, degeneracy, default'
    )

    parser.add_argument(
        '--snapshot',
        default=1,
        type=int,
        required=True,
        help='number of snapshots'
    )

    parser.add_argument(
        '--method',
        required=True,
        help='intact or partial'
    )

    parser.add_argument(
        '--node-feat',
        default='label',
        required=True,
        help='label, attr, weight, none, default'
    )
    
    args = parser.parse_args()

    if args.edge_standard == 'curvature' and args.dataset == 'COLLAB':
        print('COLLAB curvature too slow!')
        exit(0)
    
    if args.dataset == 'COLLAB' and args.method == 'intact' and int(args.snapshot) > 2:
        print('COLLAB intact > 2')
        exit(0)
    
    if args.dataset == 'COLLAB' and args.method == 'partial' and int(args.snapshot) > 10:
        print('COLLAB partial > 10')
        exit(0)

    if args.edge_standard == 'random10' and args.method == 'intact' and int(args.snapshot) > 2:
        print('random10 intact continue!')
        exit(0)

    # FIXME !!!!
    local = False

    if args.node_feat == 'default':
        args.node_feat = dataset_config[args.dataset]['node-feat']

    if args.edge_standard == 'default':
        args.edge_standard = dataset_config[args.dataset]['edge-standard']
    assert args.node_feat in reweight_node_method
    assert args.edge_standard in reweight_edge_method


    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    filtrated_path = osp.join(path, args.dataset, 'filtrated')
    processed_path = osp.join(path, args.dataset, 'processed')

    result_filename = osp.join(filtrated_path,'{}-{}-snapshot-{:0>3}.pt'.format(args.edge_standard,args.method,args.snapshot))
    
    if osp.exists(result_filename):
        print(f'{result_filename} exist! continue!')
        exit(0)

    
    if osp.exists(osp.join(processed_path, 'graphs-{}.pt'.format(args.edge_standard))):
        graphs = torch.load(osp.join(processed_path, 'graphs-{}.pt'.format(args.edge_standard)))
    else:
        graphs = load_tu_graphs(args.dataset)
        reweight_edge_method[args.edge_standard](graphs)
        if not osp.exists(processed_path):
            os.makedirs(processed_path,exists=True)
        torch.save(graphs,osp.join(processed_path, 'graphs-{}.pt'.format(args.edge_standard)))
    reweight_node_method[args.node_feat](graphs)

    y = [graph['label'] for graph in graphs]
    enc = sklearn.preprocessing.LabelEncoder()
    enc = enc.fit(y)
    y = enc.transform(y)

    print(args)




    sum_edges = 0
    sum_nodes = 0
    for graph in graphs:
        sum_edges += graph.ecount()
        sum_nodes += graph.vcount()
    if sum_edges / len(graphs) < args.snapshot:
        print('too much snapshot!')
        exit(0)

   
    if not osp.exists(filtrated_path):
        os.makedirs(filtrated_path, exist_ok=True)

    args.edge_standard = 'weight'
    edge_attr_collect = []
    for graph in graphs:
        edge_attr_collect.extend(graph.es[args.edge_standard])
    edge_attr_collect = sorted(list(set(edge_attr_collect)))
    print(args.snapshot, len(edge_attr_collect))

    if args.snapshot > len(edge_attr_collect):
        print('too much snapshot!')
        exit(0)

    snapshot_list = [-1e9]
    for i in range(args.snapshot):
        snapshot_list.append(
            edge_attr_collect[(i + 1) * len(edge_attr_collect) // args.snapshot - 1])

    tic = time.process_time()
    new_sum_nodes = 0
    new_sum_edges = 0
    
    for i, graph in enumerate(graphs):
        graph['label'] = y[i]
    
    filtrated_snapshots = [build_filtrated_snapshots(graph, args.method, snapshot_list) for graph in graphs]

    # print(filtrated_snapshots, filtrated_snapshots[0], filtrated_snapshots[0][0])

    for snapshots in filtrated_snapshots:
        for snapshot in snapshots:
            new_sum_nodes += snapshot.vcount()
            new_sum_edges += snapshot.ecount()

    torch.save(filtrated_snapshots,result_filename)
    
    print(f'save {result_filename}!')
    print(f'[{args.method} snapshot mode] rebuild graph time: {time.process_time()-tic:.2f}(s)')
    print('#node {:.2f} ==> {:.2f}'.format(
        sum_nodes / len(graphs), new_sum_nodes / len(graphs) / int(args.snapshot)))
    print('#edge {:.2f} ==> {:.2f}'.format(
        sum_edges / len(graphs), new_sum_edges / len(graphs) / int(args.snapshot)))
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

def build_filtrated_graph_with_snapshot(graph: ig.Graph, method, snapshot_list: List, local: bool):
    if len(snapshot_list) == 2:
        graph.vs['feat'] = graph.vs['weight']
        graph.vs['layer'] = [0 for _ in range(graph.vcount())]
        return graph
    global link_edge
    assert 'weight' in graph.es.attributes()
    assert 'weight' in graph.vs.attributes()
    
    if local == False:
        assert len(snapshot_list) > 0

    snapshot = len(snapshot_list) - 1
    n_nodes = graph.vcount()
    n_edges = graph.ecount()

    edge_weights = np.array(graph.es['weight'])
    edge_indices = np.argsort(edge_weights, kind='stable')

    # print(snapshot_list, sorted(edge_weights))

    vid_cnt, vid_map = 0, {}
    for snap in range(snapshot):
        if local:
            if method == 'intact':
                idx = range(0, n_edges * (snap + 1) // snapshot)
            else:
                idx = range(n_edges * snap // snapshot,
                            n_edges * (snap + 1) // snapshot)
        else:
            filter_value_l = snapshot_list[snap]
            filter_value_r = snapshot_list[snap + 1]
            idx_l, idx_r = find_interval(edge_weights[edge_indices], filter_value_l, filter_value_r)
            # print(idx_l, idx_r, filter_value_l, filter_value_r, edge_weights[edge_indices][idx_l], edge_weights[edge_indices][idx_r - 1])

            if method == 'intact':
                idx = range(0, idx_r)
            else:
                idx = range(idx_l, idx_r)

        for i, (edge_index, edge_weight) in enumerate(zip(edge_indices[idx], edge_weights[edge_indices[idx]])):
            u, v = graph.es[edge_index].tuple
            if (snap, u) not in vid_map:
                vid_map[(snap, u)] = vid_cnt
                vid_cnt += 1
            if (snap, v) not in vid_map:
                vid_map[(snap, v)] = vid_cnt
                vid_cnt += 1
        if local == False and idx_r == len(edge_weights):
            break
    
    degrees = graph.degree()
    for v, d in enumerate(degrees):
        if d == 0:
            vid_map[(0, v)] = vid_cnt
            vid_cnt += 1

    filtrated_graph = ig.Graph(n=vid_cnt)
    node_attr_list = []
    node_layer_list = []
    for k, v in vid_map.items():
        node_attr_list.append(graph.vs[k[1]]['weight'])
        node_layer_list.append(k[0])
    filtrated_graph.vs['feat'] = node_attr_list
    filtrated_graph.vs['layer'] = node_layer_list

    last_upd_nodes = {}
    for i in range(n_nodes):
        last_upd_nodes[i] = -1

    for snap in range(snapshot):
        if local:
            if method == 'intact':
                idx = range(0, n_edges * (snap + 1) // snapshot)
            else:
                idx = range(n_edges * snap // snapshot,
                            n_edges * (snap + 1) // snapshot)
        else:
            filter_value_l = snapshot_list[snap]
            filter_value_r = snapshot_list[snap + 1]
            idx_l, idx_r = find_interval(edge_weights[edge_indices], filter_value_l, filter_value_r)

            if method == 'intact':
                idx = range(0, idx_r)
            else:
                idx = range(idx_l, idx_r)

        upd_nodes = set()

        for i, (edge_index, edge_weight) in enumerate(zip(edge_indices[idx], edge_weights[edge_indices][idx])):
            u, v = graph.es[edge_index].tuple
            upd_nodes.add(u)
            upd_nodes.add(v)
            assert node_layer_list[vid_map[(snap, u)]] == node_layer_list[vid_map[(snap, v)]]
            filtrated_graph.add_edge(vid_map[(snap, u)], vid_map[(snap, v)], weight=edge_weight)

        if USE_LINK_EDGE:
            for v in upd_nodes:
                if last_upd_nodes[v] != -1:
                    link_edge += 1
                    filtrated_graph.add_edge(last_upd_nodes[v], vid_map[(snap, v)], weight=LINK_EDGE_WEIGHT)
                    assert node_layer_list[last_upd_nodes[v]] != node_layer_list[vid_map[(snap, v)]]
                last_upd_nodes[v] = vid_map[(snap, v)]

        if local == False and idx_r == len(edge_weights):
            break
    filtrated_graph['label'] = graph['label']
    return filtrated_graph

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='MUTAG',
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
        '--node-feat',
        default='label',
        help='label, attr, weight, none, default'
    )
    
    parser.add_argument(
        '--method',
        default='intact',
        help='intact, partial'
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

    # FIXME !!!!
    local = False

    if args.node_feat == 'default':
        args.node_feat = dataset_config[args.dataset]['node-feat']

    if args.edge_standard == 'default':
        args.edge_standard = dataset_config[args.dataset]['edge-standard']
    assert args.node_feat in reweight_node_method
    assert args.edge_standard in reweight_edge_method

    if args.edge_standard == 'random10' and args.method == 'intact' and int(args.snapshot) > 2:
        print('random10 intact continue!')
        exit(0)


    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    filtrated_path = osp.join(path, args.dataset, 'filtrated')
    processed_path = osp.join(path, args.dataset, 'processed')

    result_filename = osp.join(filtrated_path,'{}-{}-{:0>3}.pt'.format(args.edge_standard,args.method,args.snapshot))
    
    if osp.exists(result_filename):
        print('{}-{}-{:0>3}.pt exist! continue!'.format(args.edge_standard,args.method,args.snapshot))
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
    # print(edge_attr_collect)
    if args.snapshot > len(edge_attr_collect):
        print('too much snapshot!')
        exit(0)

    snapshot_list = [-1e9]
    for i in range(args.snapshot):
        snapshot_list.append(
            edge_attr_collect[(i + 1) * len(edge_attr_collect) // args.snapshot - 1])

    if len(edge_attr_collect) < 100:
        weight_map = {}
        for graph in graphs:
            weight = graph.es['weight']
            for x in weight:
                if x not in weight_map:
                    weight_map[x] = 1
                else:
                    weight_map[x] += 1
        print(weight_map)

    tic = time.process_time()
    new_sum_nodes = 0
    new_sum_edges = 0
    # print(graphs[0])
    graphs = [build_filtrated_graph_with_snapshot(
            graph,
            method=args.method,
            snapshot_list=snapshot_list,
            local = local) for graph in graphs]
    # print(graphs[0])

    for i, graph in enumerate(graphs):
        graph['label'] = y[i]
        t_node, t_edge = graph.vcount(), graph.ecount()
        new_sum_nodes += graph.vcount()
        new_sum_edges += graph.ecount()
    torch.save(graphs,result_filename)
    
    print('save {}!'.format(result_filename))
    print('[{} mode] rebuild graph time: {:.2f}(s)'.format('local' if local else 'global', time.process_time() - tic))
    print('#node {:.2f} ==> {:.2f}'.format(
        sum_nodes / len(graphs), new_sum_nodes / len(graphs)))
    print('#edge {:.2f} ==> {:.2f}'.format(
        sum_edges / len(graphs), new_sum_edges / len(graphs)))
    print('#link edge: {:.2f}'.format(link_edge / len(graphs)))

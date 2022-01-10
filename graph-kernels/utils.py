from typing import List
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
from networkx.classes.function import degree
import numpy as np
import os
import glob
import itertools
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import bisect
import random
import math

DIRECT_LINK_EDGE = False

SNAPSHOT_OVERLAP = False 

LINK_EDGE_WEIGHT = 1

def find_interval(weights, filter_value_l, filter_value_r):
    if SNAPSHOT_OVERLAP:
        idx_l = lower_bound(weights, filter_value_l)
        idx_r = upper_bound(weights, filter_value_r)
        if idx_r < len(weights) and weights[idx_r] == filter_value_r:
            idx_r += 1
        return idx_l, idx_r
    else:
        idx_l = upper_bound(weights, filter_value_l)
        idx_r = upper_bound(weights, filter_value_r)
        if idx_r < len(weights) and weights[idx_r] == filter_value_r:
            idx_r += 1
        return idx_l, idx_r

def lower_bound(arr: np.array, x):
    return bisect.bisect_left(list(arr), x, lo=0, hi=len(list(arr)))

def upper_bound(arr: np.array, x):
    return bisect.bisect_right(list(arr), x, lo=0, hi=len(list(arr)))

def make_kernel_ig(graph: ig.Graph, enc):
    link_edge = 0
    assert 'feat' in graph.vs.attributes()
    assert 'layer' in graph.vs.attributes()
    new_graph = ig.Graph(n=graph.vcount())
    feats = list(zip(graph.vs['feat'], graph.vs['layer']))
    feats = [y * 1000000000 + x for x, y in feats]
    feats = enc.transform(feats)
    new_graph.vs['label'] = feats

    exist = set()
    for e in graph.es:
        u, v = e.tuple
        if u == v or (u, v) in exist or (v, u) in exist:
            continue
        exist.add((u, v))
        lu = graph.vs['layer'][u]
        lv = graph.vs['layer'][v]
        if lu != lv:
            link_edge += 1
        new_graph.add_edge(u, v)
        exist.add((u, v))
    return new_graph


def ig_to_gk_intact(graph: ig.Graph, use_layer_label=True, native_edge_weight=False) -> List:
    link_edge = 0
    assert 'feat' in graph.vs.attributes()
    assert 'layer' in graph.vs.attributes()
    bi_edge_list = []
    edge_feat = {}
    exist = set()
    for e in graph.es:
        u, v = e.tuple
        if u == v or (u, v) in exist or (v, u) in exist:
            continue
        exist.add((u, v))
        lu = graph.vs['layer'][u]
        lv = graph.vs['layer'][v]
        w = math.sqrt(np.linalg.norm(graph.vs['feat'][u] - graph.vs['feat'][v])) + LINK_EDGE_WEIGHT \
            if (native_edge_weight and lu == lv) else LINK_EDGE_WEIGHT
        if lu != lv:
            link_edge += 1
        if DIRECT_LINK_EDGE:
            if lu <= lv:
                bi_edge_list.append((u, v))
                edge_feat[(u, v)] = w
            if lv <= lu:
                bi_edge_list.append((v, u))
                edge_feat[(v, u)] = w
        else:
            bi_edge_list.append((u, v))
            bi_edge_list.append((v, u))
            edge_feat[(u, v)] = w
            edge_feat[(v, u)] = w

    if isinstance(graph.vs['feat'][0], List):
        node_feat = {i:int(v['layer']) for i, v in enumerate(graph.vs)}
    else:
        if use_layer_label:
            node_feat = {i: (int(v['feat']), int(v['layer'])) for i, v in enumerate(graph.vs)}
        else:
            node_feat = {i: int(v['feat']) for i, v in enumerate(graph.vs)}
    return [bi_edge_list, node_feat, edge_feat], link_edge
    
       
def ig_to_nx_only_edges(graph: ig.Graph) -> nx.Graph:
    edge_list = graph.get_edgelist()
    G = nx.Graph(edge_list)
    return (G)


def ig_to_nx_intact(graph: ig.Graph) -> nx.Graph:
    edge_list = graph.get_edgelist()
    G = nx.Graph(edge_list)
    for v_attr in graph.vs.attributes():
        if v_attr == '_nx_name':
            continue
        attr_dict = {}
        for i, x in enumerate(graph.vs[v_attr]):
            attr_dict[i] = x
        nx.set_node_attributes(G, attr_dict, name=v_attr)
    for e_attr in graph.es.attributes():
        attr_dict = {}
        for e in graph.es:
            u, v = e.tuple
            val = e[e_attr]
            attr_dict[(u, v)] = val
        nx.set_edge_attributes(G, attr_dict, name=e_attr)
    return G


def calc_common_neighbors(graph: ig.Graph) -> List[int]:
    nx_graph = ig_to_nx_only_edges(graph)
    edges = [e.tuple for e in graph.es]
    ans = [len(list(nx.common_neighbors(nx_graph, e[0], e[1]))) for e in edges]
    return ans


def calc_jaccard_similarity(graph: ig.Graph) -> List[float]:
    nx_graph = ig_to_nx_only_edges(graph)
    edges = [e.tuple for e in graph.es]
    ans = []
    for u, v in edges:
        neighbors_u = set(nx.neighbors(nx_graph, u))
        neighbors_v = set(nx.neighbors(nx_graph, v))
        jaccard_sim = len(neighbors_u.intersection(neighbors_v)) / len(neighbors_u.union(neighbors_v))
        ans.append(jaccard_sim)
    return ans


def calc_ccpa(graph: ig.Graph) -> List[float]:
    nx_graph = ig_to_nx_only_edges(graph)
    edges = [e.tuple for e in graph.es]
    ans = nx.common_neighbor_centrality(nx_graph, edges)
    ans = [x[2] for x in ans]
    return ans


if __name__ == '__main__':


    pass
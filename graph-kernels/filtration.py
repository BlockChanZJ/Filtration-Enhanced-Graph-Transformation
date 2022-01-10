import copy
import sys
import time
import random

import igraph as ig
import networkx as nx
from networkx.classes.function import common_neighbors
import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from tqdm import tqdm
from typing import List
from dataset import load_tu_graphs
from utils import *


def reweight_edges_with_curvature(graphs: List[ig.Graph]):
    """ relables edges with O.F. curvature. Note this requires
    converting to networkx and back. Currently this is simple since we
    have no node or edge labels. But would need to reconsider if we
    extend this to labeled graphs. """

    tic = time.time()
    for graph in graphs:
        nx_graph = ig_to_nx_only_edges(copy.deepcopy(graph))
        orc = OllivierRicci(nx_graph, alpha=0.5, verbose="INFO", proc=6)
        orc.compute_ricci_curvature()
        es_weight = [0 for _ in range(graph.ecount())]
        for u, v, e in nx_graph.edges(data=True):
            eid = graph.get_eid(v1=u, v2=v)
            es_weight[eid] = orc.G[u][v]["ricciCurvature"]
        graph.es['weight'] = es_weight
    print('calc curvature time: {:.2f}(s)'.format(
        time.time() - tic), file=sys.stdout)
    return (graphs)


def reweight_edges_with_degeneracy(graphs: List[ig.Graph]):
    tic = time.time()
    for graph in graphs:
        nx_graph = ig_to_nx_only_edges(copy.deepcopy(graph))
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
        core_numbers = nx.core_number(nx_graph)
        weights = []
        for e in graph.es:
            u, v = e.tuple
            cu = core_numbers[u] if u in core_numbers else 0
            cv = core_numbers[v] if v in core_numbers else 0
            # weights.append(max(cu, cv))
            weights.append((cu + cv) / 2)
        graph.es['weight'] = weights
    print('calc degeneracy time: {:.2f}(s)'.format(
        time.time() - tic), file=sys.stdout)
    return (graphs)


def reweight_edges_with_ccpa(graphs: List[ig.Graph]):
    tic = time.time()
    for graph in graphs:
        graph.es['weight'] = calc_ccpa(graph)
    print('calc ccpa time: {:.2f}(s)'.format(
        time.time() - tic), file=sys.stdout)
    return graphs

def reweight_edges_with_label_distance(graphs: List[ig.Graph]):
    from scipy.stats import wasserstein_distance
    tic = time.time()
    assert 'label' in graphs[0].vs.attributes()
    for graph in graphs:
        labels = graph.vs['label']
        edge_weights = []
        for e in graph.es:
            u, v = e.tuple
            u_distribution, v_distribution = [labels[u]], [labels[v]]
            for x in graph.neighbors(vertex=u):
                u_distribution.append(labels[x])
            for x in graph.neighbors(vertex=v):
                v_distribution.append(labels[x])
            dist = wasserstein_distance(u_distribution, v_distribution)
            edge_weights.append(dist)
        graph.es['weight'] = edge_weights

    print('calc label distance time: {:.2f}(s)'.format(time.time() - tic))

def reweight_edges_with_overlap(graphs: List[ig.Graph], lamda=2):
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
            edge_weights.append(common_edges * math.pow(cns, lamda) / (cns * (cns - 1)))
        graph.es['weight'] = edge_weights

    print('calc overlap time: {:.2f}(s)'.format(time.time() - tic))


def reweight_edges_with_common_neighbors(graphs: List[ig.Graph]):
    tic = time.time()
    for graph in graphs:
        graph.es['weight'] = calc_common_neighbors(graph)
    print('calc common neighbors time: {:.2f}(s)'.format(
        time.time() - tic), file=sys.stdout)
    return graphs


def reweight_edge_with_jaccard(graphs: List[ig.Graph]):
    tic = time.time()
    for graph in graphs:
        graph.es['weight'] = calc_jaccard_similarity(graph)
    print('calc jaccard time: {:.2f}(s)'.format(
        time.time() - tic), file=sys.stdout)
    return graphs


def reweight_edge_with_jaccard_with_noise(graphs: List[ig.Graph]):
    np.random.seed(42)

    tic = time.time()
    for graph in graphs:
        weights = calc_jaccard_similarity(graph)
        noise = np.random.normal(0, 0.1, size=len(weights))
        weights = (np.array(weights) + noise).tolist()
        graph.es['weight'] = weights
    print('calc jaccard with noise time: {:.2f}(s)'.format(
        time.time() - tic), file=sys.stdout)
    return graphs


def reweight_edges_with_label(graphs: List[ig.Graph]):
    assert 'label' in graphs[0].es.attributes()
    for idx, graph in enumerate(graphs):
        graph.es['weight'] = graph.es['label']


def reweight_edges_with_attribute(graphs: List[ig.Graph]):
    assert 'attr' in graphs[0].es.attributes()
    for idx, graph in enumerate(graphs):
        graph.es['weight'] = graph.es['attr']


def reweight_edges_with_node_attribute(graphs: List[ig.Graph]):
    assert 'attr' in graphs[0].vs.attributes()
    for idx, graph in enumerate(graphs):
        weights = []
        for e in graph.es:
            u, v = e.tuple
            weights.append(np.linalg.norm(np.array(graph.vs['attr'][u]) - np.array(graph.vs['attr'][v])))
        graph.es['weight'] = weights


def reweight_node_with_degree(graphs: List[ig.Graph]):
    for idx, graph in enumerate(graphs):
        graph.vs['weight'] = [v.degree() for v in graph.vs]
    return (graphs)


def reweight_node_with_degeneracy(graphs: List[ig.Graph]):
    for idx, graph in enumerate(graphs):
        nx_graph = ig_to_nx_only_edges(copy.deepcopy(graph))
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
        core_numbers = nx.core_number(nx_graph)
        graph.vs['weight'] = [core_numbers[v['_nx_name']]
                              if v['_nx_name'] in core_numbers else 0 for v in graph.vs]
    return graphs


def reweight_node_with_label(graphs: List[ig.Graph]):
    assert 'label' in graphs[0].vs.attributes()
    for idx, graph in enumerate(graphs):
        graph.vs['weight'] = graph.vs['label']


def reweight_node_with_attribute(graphs: List[ig.Graph]):
    assert 'attr' in graphs[0].vs.attributes()
    for idx, graph in enumerate(graphs):
        graph.vs['weight'] = graph.vs['attr']

def reweight_edge_with_random10(graphs: List[ig.Graph]):
    random.seed(123)
    for idx, graph in enumerate(graphs):
        graph.es['weight'] = [random.randint(
            0, 10) for _ in range(graph.ecount())]

def reweight_edge_with_random100(graphs: List[ig.Graph]):
    random.seed(123)
    for idx, graph in enumerate(graphs):
        graph.es['weight'] = [random.randint(
            0, 100) for _ in range(graph.ecount())]


def reweight_edge_with_random1000(graphs: List[ig.Graph]):
    random.seed(123)
    for idx, graph in enumerate(graphs):
        graph.es['weight'] = [random.randint(
            0, 1000) for _ in range(graph.ecount())]


def reweight_edge_with_random10000(graphs: List[ig.Graph]):
    random.seed(123)
    for idx, graph in enumerate(graphs):
        graph.es['weight'] = [random.randint(
            0, 10000) for _ in range(graph.ecount())]


def reweight_node_with_none(graphs: List[ig.Graph]):
    for idx, graph in enumerate(graphs):
        graph.vs['weight'] = [0 for _ in range(graph.vcount())]


reweight_node_method = {
    'label': reweight_node_with_label,
    'attr': reweight_node_with_attribute,
    'degree': reweight_node_with_degree,
    'degeneracy': reweight_node_with_degeneracy,
    'none': reweight_node_with_none,
}
reweight_edge_method = {
    'label': reweight_edges_with_label,
    'attr': reweight_edges_with_attribute,
    'curvature': reweight_edges_with_curvature,
    'vattr': reweight_edges_with_node_attribute,
    'random10': reweight_edge_with_random10,
    'random100': reweight_edge_with_random100,
    'random1000': reweight_edge_with_random1000,
    'random10000': reweight_edge_with_random10000,
    'cn': reweight_edges_with_common_neighbors,
    'ccpa': reweight_edges_with_ccpa,
    'degeneracy': reweight_edges_with_degeneracy,
    'jaccard': reweight_edge_with_jaccard,
    'jaccard-noise': reweight_edge_with_jaccard_with_noise,
    'ld': reweight_edges_with_label_distance,
    'overlap':reweight_edges_with_overlap,
}

if __name__ == '__main__':
    # graphs = load_tu_graphs('MUTAG')
    # graphs = reweight_node_with_degeneracy(graphs)
    # print(graphs[0])
    pass

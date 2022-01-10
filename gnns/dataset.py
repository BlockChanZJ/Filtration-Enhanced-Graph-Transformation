import glob
import os
import shutil
import zipfile

import igraph as ig
import os.path as osp

import networkx as nx
import requests
from tqdm import tqdm
from typing import List
from config import dataset_config
import torch

url_pref = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/'


def download_and_extract(ds_name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    raw_path = osp.join(path, ds_name, 'raw')
    url = url_pref + '/' + ds_name + '.zip'
    response = requests.get(url, stream=True)
    zip_file = osp.join(path, url.split('/')[-1])
    with open(zip_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    f = zipfile.ZipFile(zip_file, 'r')
    for file in f.namelist():
        f.extract(file, path)
    for file in f.namelist()[1:]:
        print(osp.join(path, file), osp.join(raw_path, file.split('/')[-1]))
        shutil.move(osp.join(path, file), osp.join(raw_path, file.split('/')[-1]))
    os.remove(osp.join(path, ds_name + '.zip'))


def process_digraph(ds_name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    raw_path = osp.join(path, ds_name, 'raw')
    with open(osp.join(raw_path, ds_name + '_A.txt'), 'r') as f:
        lines = f.readlines()
        if ds_name == 'NCI1' or ds_name == 'NCI109':
            adj = [list(map(int, line.split(','))) for line in lines]
        else:
            adj = [list(map(int, line.split(', '))) for line in lines]
        adj = [line if line[0] < line[1] else [line[1], line[0]] for i, line in enumerate(adj)]

    filename = osp.join(raw_path, ds_name + '_edge_labels.txt')
    if osp.exists(filename):
        with open(filename, 'r') as f:
            edge_label_lines = f.readlines()
    else:
        edge_label_lines = None

    filename = osp.join(raw_path, ds_name + '_edge_attributes.txt')
    if osp.exists(filename):
        with open(filename, 'r') as f:
            edge_attr_lines = f.readlines()
    else:
        edge_attr_lines = None
    # exit(0)

    if edge_label_lines is not None and edge_attr_lines is not None:
        tuples = sorted(zip(adj, edge_label_lines, edge_attr_lines), key=lambda x: (x[0][0], x[0][1]))
        with open(osp.join(raw_path, ds_name + '_A.txt'), 'w') as f1:
            with open(osp.join(raw_path, ds_name + '_edge_labels.txt'), 'w') as f2:
                with open(osp.join(raw_path, ds_name + '_edge_attributes.txt'), 'w') as f3:
                    for i, (x, y, z) in enumerate(tuples):
                        if i > 0 and x == tuples[i - 1][0]:
                            continue
                        f1.write('{}, {}\n'.format(x[0], x[1]))
                        f2.write('{}'.format(y))
                        f3.write('{}'.format(z))
    elif edge_label_lines is not None:
        tuples = sorted(zip(adj, edge_label_lines), key=lambda x: (x[0][0], x[0][1]))
        with open(osp.join(raw_path, ds_name + '_A.txt'), 'w') as f1:
            with open(osp.join(raw_path, ds_name + '_edge_labels.txt'), 'w') as f2:
                for i, (x, y) in enumerate(tuples):
                    if i > 0 and x == tuples[i - 1][0]:
                        continue
                    f1.write('{}, {}\n'.format(x[0], x[1]))
                    f2.write('{}'.format(y))
    elif edge_attr_lines is not None:
        tuples = sorted(zip(adj, edge_attr_lines), key=lambda x: (x[0][0], x[0][1]))
        with open(osp.join(raw_path, ds_name + '_A.txt'), 'w') as f1:
            with open(osp.join(raw_path, ds_name + '_edge_attributes.txt'), 'w') as f3:
                for i, (x, z) in enumerate(tuples):
                    if i > 0 and x == tuples[i - 1][0]:
                        continue
                    f1.write('{}, {}\n'.format(x[0], x[1]))
                    f3.write('{}'.format(z))
    else:
        tuples = sorted(adj, key=lambda x: (x[0], x[1]))
        # print(tuples)
        with open(osp.join(raw_path, ds_name + '_A.txt'), 'w') as f1:
            for i, (x) in enumerate(tuples):
                if i > 0 and x == tuples[i - 1]:
                    continue
                f1.write('{}, {}\n'.format(x[0], x[1]))


def process_tu_dataset(ds_name='MUTAG'):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    raw_path = osp.join(path, ds_name, 'raw')
    processed_path = osp.join(path, ds_name, 'processed')
    if not osp.exists(raw_path):
        os.makedirs(raw_path, exist_ok=True)
    if not osp.exists(processed_path):
        os.makedirs(processed_path, exist_ok=True)
    download_and_extract(ds_name)
    process_digraph(ds_name)
    with open(osp.join(raw_path, ds_name + '_graph_indicator.txt'), 'r') as f:
        node_indicator = list(map(int, f.readlines()))

    # create igraph
    n_graphs = max(node_indicator)

    # create node_indicator_map
    node_indicator_map = {}
    for i, x in enumerate(node_indicator):
        node_indicator_map[i] = x - 1

    # create node_base_map
    node_base_map = {}
    sum, last_gid = 0, -1
    for x in node_indicator:
        x = x - 1
        if last_gid != x:
            last_gid = x
            node_base_map[x] = sum
        sum += 1
    node_base_map[last_gid + 1] = sum

    with open(osp.join(raw_path, ds_name + '_graph_labels.txt'), 'r') as f:
        graph_label = list(map(int, f.readlines()))

    # construct graphs
    edges = [[] for _ in range(n_graphs)]
    es_attrs = [[] for _ in range(n_graphs)]
    es_labels = [[] for _ in range(n_graphs)]
    vs_labels = [[] for _ in range(n_graphs)]
    vs_attrs = [[] for _ in range(n_graphs)]

    with open(osp.join(raw_path, ds_name + '_A.txt'), 'r') as f:
        adj_lines = f.readlines()

    filename = osp.join(raw_path, ds_name + '_edge_labels.txt')
    if osp.exists(filename) and dataset_config[ds_name]['e-label']:
        with open(filename, 'r') as f:
            edge_label_lines = f.readlines()
    else:
        edge_label_lines = None

    filename = osp.join(raw_path, ds_name + '_edge_attributes.txt')
    if osp.exists(filename) and dataset_config[ds_name]['e-attr']:
        with open(filename, 'r') as f:
            edge_attr_lines = f.readlines()
    else:
        edge_attr_lines = None

    filename = osp.join(raw_path, ds_name + '_node_labels.txt')
    if osp.exists(filename) and dataset_config[ds_name]['v-label']:
        with open(filename, 'r') as f:
            node_label_lines = f.readlines()
            for vid, line in enumerate(node_label_lines):
                gid = node_indicator_map[vid]
                vs_labels[gid].append((int)(line))

    filename = osp.join(raw_path, ds_name + '_node_attributes.txt')
    if osp.exists(filename) and dataset_config[ds_name]['v-attr']:
        with open(filename, 'r') as f:
            node_attr_lines = f.readlines()
            for vid, line in enumerate(node_attr_lines):
                gid = node_indicator_map[vid]
                if ds_name == 'ENZYMES' or ds_name == 'BZR' or ds_name == 'COX2' or ds_name == 'DHFR' or ds_name == 'PROTEINS_full':
                    line = line.strip('\n').strip(' ')
                    line = line.split(',')
                    line = [float(x.strip(' ')) for x in line]
                    vs_attrs[gid].append(line)
                else:
                    vs_attrs[gid].append((float)(line))

    # construct graphs
    for lid, line in enumerate(adj_lines):
        if edge_attr_lines is not None:
            attr = float(edge_attr_lines[lid].strip('\n'))
        else:
            attr = None

        if edge_label_lines is not None:
            label = int(edge_label_lines[lid].strip('\n'))
        else:
            label = None

        x, y = map(int, line.split(', '))
        x -= 1
        y -= 1
        assert node_indicator_map[x] == node_indicator_map[y]
        gid = node_indicator_map[x]
        assert x >= node_base_map[gid] and x < node_base_map[gid + 1]
        assert y >= node_base_map[gid] and y < node_base_map[gid + 1]
        x = x - node_base_map[gid]
        y = y - node_base_map[gid]
        edges[gid].append((x, y))
        if attr is not None:
            es_attrs[gid].append(attr)
        if label is not None:
            es_labels[gid].append(label)

    def get_even_lines(x):
        return x
        # assert len(x) % 2 == 0
        # ans = []
        # for i, y in enumerate(x):
        #     if i % 2 == 0:
        #         ans.append(y)
        #     else:
        #         # print(y, x[i - 1])
        #         if isinstance(y, tuple):
        #             assert (y[1], y[0]) == x[i - 1]
        # return ans

    # add graph label in igraph
    assert len(graph_label) == n_graphs
    new_graphs = []
    for gid in range(n_graphs):
        graph = ig.Graph()
        graph.add_vertices(node_base_map[gid + 1] - node_base_map[gid])
        graph.add_edges(get_even_lines(edges[gid]))

        graph['label'] = graph_label[gid]
        if len(vs_labels[gid]) > 0:
            graph.vs['label'] = vs_labels[gid]
        if len(vs_attrs[gid]) > 0:
            graph.vs['attr'] = vs_attrs[gid]
        if len(es_labels[gid]) > 0:
            graph.es['label'] = get_even_lines(es_labels[gid])
        if len(es_attrs[gid]) > 0:
            graph.es['attr'] = get_even_lines(es_attrs[gid])
        graph.vs['_nx_name'] = list(range(node_base_map[gid + 1] - node_base_map[gid]))
        new_graphs.append(graph)
    torch.save(new_graphs, osp.join(processed_path, 'graphs.pt'))


def load_tu_graphs(ds_name='MUTAG') -> List[ig.Graph]:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', ds_name, 'processed')
    if not osp.exists(path + '/graphs.pt'):
        process_tu_dataset(ds_name)
    graphs = torch.load(path + '/graphs.pt')
    return graphs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='dataset'
    )

    args = parser.parse_args()
    process_tu_dataset(ds_name=args.dataset)
    # graphs = load_tu_graphs(ds_name=args.dataset)
    # print(graphs[0])

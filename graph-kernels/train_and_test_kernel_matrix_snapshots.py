import json
import math
import os
import sys
import time
from warnings import catch_warnings
from grakel.kernels.vertex_histogram import VertexHistogram
from networkx.readwrite.nx_shp import edges_from_line

import numpy as np
from numpy.linalg import pinv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC
import argparse
import os.path as osp
import glob
from torch.jit import Error
from tqdm import tqdm
import igraph as ig
import torch
from config import dataset_config
import grakel
from utils import *
import random
import math


EPS = 1e-5

def check_semi_positive_definite(mat):
    return np.all(np.linalg.eigvals(mat) > -EPS)

def normalize_gram_matrix(gram_matrix):
    gram_matrix[np.isnan(gram_matrix)] = 0
    n = gram_matrix.shape[0]
    gram_matrix_norm = np.zeros([n, n], dtype=np.float64)


    for i in range(0, n):
        for j in range(i, n):
            if not (math.fabs(gram_matrix[i][i]) < EPS or math.fabs(gram_matrix[j][j]) < EPS):
                g = gram_matrix[i][j] / \
                    math.sqrt(math.fabs(gram_matrix[i][i] * gram_matrix[j][j]))
                gram_matrix_norm[i][j] = g
                gram_matrix_norm[j][i] = g

    gram_matrix_norm[np.isnan(gram_matrix_norm)] = 0

    if not check_semi_positive_definite(gram_matrix_norm):
        print('check semi-pos-definite error!')
        exit(-1)

    return gram_matrix_norm


# 10-CV for kernel svm and hyperparameter selection.
def svm_evaluation(mat, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]):
    # Acc. over all repetitions.
    accuracy_all = []

    for i in tqdm(range(num_repetitions),desc='run svm'):
    # for i in range(num_repetitions):
        # Test acc. over all folds.
        kf = KFold(n_splits=10, shuffle=True, random_state=42 + i)
        accuracy = []
        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Determine hyperparameters
            train = mat[train_index, :]
            train = train[:, train_index]
            test = mat[test_index, :]
            test = test[:, train_index]
            c_train = classes[train_index]
            c_test = classes[test_index]

            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            clf = GridSearchCV(
                SVC(kernel='precomputed'), params, cv=5, scoring='accuracy', verbose=0, n_jobs=1)
            clf.fit(train, c_train)
            c_pred = clf.predict(test)
            best_test_acc = accuracy_score(c_test, c_pred) * 100.0
            accuracy.append(best_test_acc)
        accuracy_all.append(np.array(accuracy).mean())

    return np.array(accuracy_all).mean(), np.array(accuracy_all).std()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='MUTAG',
        help='dataset'
    )
    parser.add_argument(
        '--snapshot',
        required=True,
        help='001 002 ...'
    )
    parser.add_argument(
        '--method',
        required=True,
        help='intact, partial'
    )
    parser.add_argument(
        '--edge-standard',
        required=True,
    )
    parser.add_argument(
        '--dir',
        required=True
    )

    # fix random seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = parser.parse_args()
    dir = args.dir

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    filtrated_path = osp.join(path, args.dataset, 'filtrated')

    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '..', 'data', args.dataset, 'filtrated')

    if args.edge_standard == 'default':
        args.edge_standard = dataset_config[args.dataset]['edge-standard']

    filename = osp.join(path,'{}-{}-snapshot-{:0>3d}.pt'.format(args.edge_standard,args.method,int(args.snapshot)))
    if not osp.exists(filename):
        print('please build {}-{}-snapshot-{:0>3d}.pt first!'.format(args.edge_standard,args.method,int(args.snapshot)))
        exit(0)

    res_json_file = dir + f'/{args.dataset}-{args.method}-snapshot-{int(args.snapshot):0>3d}.log'
    if osp.exists(res_json_file):
        qwq = res_json_file.split('/')[-1]
        print(f'{qwq} exist! continue!')
        exit(0)
        
    ds_name = args.dataset

    # if ds_name == 'REDDIT-MULTI-5K' and args.method == 'intact' and int(args.snapshot) > 3:
    #     print('too much snapshot! continue!')
    #     exit(0)

    if (ds_name == 'BZR' or ds_name == 'ENZYMES' or ds_name == 'COX2' or ds_name == 'DHFR' or ds_name == 'PROTEINS') and args.edge_standard == 'vattr':
        native_edge_weight = True
    else:
        native_edge_weight = False
    
    native_edge_weight = False

    # change format
    graphs = torch.load(filename)
    y = [graph[0]['label'] for graph in graphs]
    # print(y)
    tmp = [[ig_to_gk_intact(snapshot,native_edge_weight=native_edge_weight)[0] for snapshot in snapshots] for snapshots in graphs]

    del graphs
    graphs = [[] for _ in range(int(args.snapshot))]
    for snapshots in tmp:
        for i, snapshot in enumerate(snapshots):
            graphs[i].append(snapshot)
    del tmp

    if not osp.exists(dir):
        os.makedirs(dir, exist_ok=True)
    result_json = {}



    def get_max(x, y):
        x, xx = x.split(' +- ')
        y, yy = y.split(' +- ')
        x,y,xx,yy = float(x), float(y), float(xx), float(yy)
        if x > y or (x == y and xx < yy):
            return str(x) + ' +- ' + str(xx)
        else:
            return str(y) + ' +- ' + str(yy)

    kernel_method = {
        # 'VH': grakel.VertexHistogram(normalize=False),
        'WL_1': grakel.WeisfeilerLehman(normalize=False, n_iter=1,),
        'WL_2': grakel.WeisfeilerLehman(normalize=False, n_iter=2),
        'WL_3': grakel.WeisfeilerLehman(normalize=False, n_iter=3),
        'WL_4': grakel.WeisfeilerLehman(normalize=False, n_iter=4),
        'WL_5': grakel.WeisfeilerLehman(normalize=False, n_iter=5),
        'WL_6': grakel.WeisfeilerLehman(normalize=False, n_iter=6),
        'WL_7': grakel.WeisfeilerLehman(normalize=False, n_iter=7),
        'GL_3': grakel.GraphletSampling(normalize=False,k=3,sampling={'n_samples':500},random_state=123),
        'GL_4': grakel.GraphletSampling(normalize=False,k=4,sampling={'n_samples':500},random_state=123),
        'GL_5': grakel.GraphletSampling(normalize=False,k=5,sampling={'n_samples':500},random_state=123),
        'SP': grakel.ShortestPath(normalize=False,algorithm_type='dijkstra'),
    }


    for k in kernel_method:
        result_json[k] = {}
    for k, func in kernel_method.items():
        tic = time.process_time()
        grk = func
        mat = np.zeros(shape=(len(y), len(y)))
        for i, snapshots in enumerate(graphs):
            tmp_mat = grk.fit_transform(snapshots)
            tmp_mat[np.isnan(tmp_mat)] = 0
            mat = mat + tmp_mat
        time_cost = time.process_time() - tic
        mat = normalize_gram_matrix(mat)
        acc, std = svm_evaluation(mat, np.array(y))
        t = k
        if 'acc' not in result_json[t]:
            result_json[t]['acc'] = '{:.2f} +- {:.2f}'.format(acc, std)
            result_json[t]['time'] = '{:.2f}(s)'.format(time_cost)
        else:
            result_json[t]['acc'] = get_max(result_json[t]['acc'],  '{:.2f} +- {:.2f}'.format(acc, std))
        print(f'{args.dataset}-{args.method}-snapshot-{int(args.snapshot)}: {k} {acc:.2f} +- {std:.2f}, {time_cost:.2f}s')
        res = json.dumps(result_json, indent=4)
        with open(dir + '/{}-{}-snapshot-{:0>3d}.log'.format(args.dataset,args.method, int(args.snapshot)), 'w') as f:
            f.write(res)


import json
import math
import os
import sys
import time
from grakel.kernels.vertex_histogram import VertexHistogram
from networkx.readwrite.nx_shp import edges_from_line

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC
import argparse
import os.path as osp
import glob
from torch.functional import norm
from tqdm import tqdm
import igraph as ig
import torch
from config import dataset_config
import grakel
from utils import *
import random

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
                SVC(kernel='precomputed'), params, cv=5, scoring='accuracy', verbose=0,n_jobs=1)
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
        '--method',
        required=True,
        help='intact or partial'
    )
    parser.add_argument(
        '--snapshot',
        required=True,
        help='001 002 ...'
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

    filename = osp.join(path,'{}-{}-{:0>3}.pt'.format(args.edge_standard,args.method,int(args.snapshot)))
    if not osp.exists(filename):
        print('please build {}-{}-{:0>3}.pt first!'.format(args.edge_standard,args.method,int(args.snapshot)))
        exit(0)

    res_json_file = dir + f'/{args.dataset}-{args.method}-{int(args.snapshot):0>3d}.log'
    if osp.exists(res_json_file):
        qwq = res_json_file.split('/')[-1]
        print(f'{qwq} exist! continue!')
        exit(0)
    
    ds_name = args.dataset
    if (ds_name == 'BZR' or ds_name == 'ENZYMES' or ds_name == 'COX2' or ds_name == 'DHFR' or ds_name == 'PROTEINS') and args.edge_standard == 'vattr':
        native_edge_weight = True
    else:
        native_edge_weight = False

    native_edge_weight = False
    
    # change format
    graphs = torch.load(filename)
    y = [graph['label'] for graph in graphs]
    graphs = [ig_to_gk_intact(graph,native_edge_weight=native_edge_weight) for graph in graphs]
    link_edge = sum([graph[1] for graph in graphs])
    graphs = [graph[0] for graph in graphs]

    # if link_edge == 0 and int(snapshot) > 1:
    #     print('link edge is 0! continue!')

    print('# link edge: {:.2f}'.format(link_edge / len(graphs)))
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
        'WLOA_1':grakel.WeisfeilerLehmanOptimalAssignment(normalize=False,n_iter=1),
        'WLOA_2':grakel.WeisfeilerLehmanOptimalAssignment(normalize=False,n_iter=2),
        'WLOA_3':grakel.WeisfeilerLehmanOptimalAssignment(normalize=False,n_iter=3),
        'WLOA_4':grakel.WeisfeilerLehmanOptimalAssignment(normalize=False,n_iter=4),
        'WLOA_5':grakel.WeisfeilerLehmanOptimalAssignment(normalize=False,n_iter=5),
        'WLOA_6':grakel.WeisfeilerLehmanOptimalAssignment(normalize=False,n_iter=6),
        'WLOA_7':grakel.WeisfeilerLehmanOptimalAssignment(normalize=False,n_iter=7),
        'PM_1':grakel.PyramidMatch(normalize=False,L=2,d=4),
        'PM_2':grakel.PyramidMatch(normalize=False,L=2,d=6),
        'PM_3':grakel.PyramidMatch(normalize=False,L=2,d=8),
        'PM_4':grakel.PyramidMatch(normalize=False,L=2,d=10),
        'PM_5':grakel.PyramidMatch(normalize=False,L=4,d=4),
        'PM_6':grakel.PyramidMatch(normalize=False,L=4,d=6),
        'PM_7':grakel.PyramidMatch(normalize=False,L=4,d=8),
        'PM_8':grakel.PyramidMatch(normalize=False,L=4,d=10),
        'PM_9':grakel.PyramidMatch(normalize=False,L=6,d=4),
        'PM_10':grakel.PyramidMatch(normalize=False,L=6,d=6),
        'PM_11':grakel.PyramidMatch(normalize=False,L=6,d=8),
        'PM_12':grakel.PyramidMatch(normalize=False,L=6,d=10),
        'MLG_1':grakel.MultiscaleLaplacian(normalize=False,random_state=123,gamma=0.1,heta=0.1),
        'MLG_2':grakel.MultiscaleLaplacian(normalize=False,random_state=123,gamma=0.01,heta=0.1),
        'MLG_3':grakel.MultiscaleLaplacian(normalize=False,random_state=123,gamma=0.1,heta=0.01),
        'MLG_4':grakel.MultiscaleLaplacian(normalize=False,random_state=123,gamma=0.01,heta=0.01),
        'CSM_3':grakel.SubgraphMatching(normalize=False,k=3),
        'CSM_4':grakel.SubgraphMatching(normalize=False,k=4),
        'CSM_5':grakel.SubgraphMatching(normalize=False,k=5),
    }


    for k in kernel_method:
        result_json[k] = {}
    for k, func in kernel_method.items():
        tic = time.process_time()
        grk = func        
        mat = grk.fit_transform(graphs)
        time_cost = time.process_time() - tic
        mat = normalize_gram_matrix(mat)
        acc, std = svm_evaluation(mat, np.array(y))
        t = k
        if (ds_name == 'COX2_MD' or ds_name == 'DHFR_MD' or ds_name == 'ER_MD') and 'CSM_' in k:
            continue
        if 'acc' not in result_json[t]:
            result_json[t]['acc'] = '{:.2f} +- {:.2f}'.format(acc, std)
            result_json[t]['time'] = '{:.2f}(s)'.format(time_cost)
        else:
            result_json[t]['acc'] = get_max(result_json[t]['acc'],  '{:.2f} +- {:.2f}'.format(acc, std))
        print(f'{args.dataset}-{args.method}-{args.snapshot}: {k} {acc:.2f} +- {std:.2f}, {time_cost:.2f}s')
        res = json.dumps(result_json, indent=4)
        with open(dir + '/{}-{}-{:0<3}.log'.format(args.dataset, args.method, args.snapshot), 'w') as f:
            f.write(res)


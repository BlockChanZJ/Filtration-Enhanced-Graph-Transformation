import json

import tabulate
import pandas as pd
import numpy as np
import os
import os.path as osp
import glob
import argparse

if __name__ == '__main__':

    filtration_methods = ['cn', 'degeneracy', 'curvature']
    filtration_methods_attr = ['attr', 'vattr']
    datasets = [
        ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI',
            'NCI1', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'],
        ['BZR', 'COX2', 'DHFR', 'ENZYMES', 'PROTEINS', 'PROTEINS_full'],
        ['BZR_MD', 'COX2_MD', 'DHFR_MD', 'ER_MD']
    ]
    kernel_methods = ['WL', 'SP', 'GL', 'PM', 'NH', 'CSM', 'MLG', 'WLOA']

    '''
    graph kernels: self-defined filtration & graphs
    '''

    def parse_name(s):
        s = s.replace('-snapshot', '')
        l = s.split('-')
        ds_name = ''
        for x in l[:-2]:
            if len(ds_name) > 0:
                ds_name = ds_name + '-'
            ds_name = ds_name + x
        return ds_name, l[-2], l[-1]

    def parse_acc(x):
        x_acc, x_std = x.split(' +- ')
        return f'{float(x_acc):.1f} +- {float(x_std):.1f}'

    def get_diff(x, y):
        x_acc, x_std = x.split(' +- ')
        y_acc, y_std = y.split(' +- ')
        return float(x_acc) - float(y_acc)

    kernel_map, method_map = {}, {}

    for filtration in filtration_methods:

        # FEG
        files = glob.glob(osp.join(f'log1-{filtration}', '*.log'))
        for file in files:
            with open(file, 'r') as f:
                res = json.load(f)
            file = file.replace(f'log1-{filtration}' + '/', '')
            file = file.strip('.log')
            ds_name, method, layer = parse_name(file)
            if ds_name not in datasets[0]:
                continue
            if method == 'intact' and int(layer) == 1:
                continue
            if ds_name not in kernel_map:
                kernel_map[ds_name] = {}
                method_map[ds_name] = {}
            for gk in res:
                gk_method = gk
                if gk.startswith('WL_'):
                    acc_entry = f'WL-{method}-FEG'
                    gk_method = 'WL'
                elif gk.startswith('GL_'):
                    acc_entry = f'GL-{method}-FEG'
                    gk_method = 'GL'
                elif gk.startswith('WLOA_'):
                    acc_entry = f'WLOA-{method}-FEG'
                    gk_method = 'WLOA'
                elif gk.startswith('CSM_'):
                    acc_entry = f'CSM-{method}-FEG'
                    gk_method = 'CSM'
                else:
                    acc_entry = f'{gk}-{method}-FEG'
                    
                
                if gk_method not in kernel_methods:
                    continue
                # print(gk)
                if 'acc' not in res[gk]:
                    continue
                if acc_entry not in kernel_map[ds_name]:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
                elif get_diff(res[gk]['acc'], kernel_map[ds_name][acc_entry]) > 0:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
        # FES
        files = glob.glob(osp.join(f'log1-{filtration}-snapshot', '*.log'))
        for file in files:
            with open(file, 'r') as f:
                res = json.load(f)
            file = file.replace(f'log1-{filtration}-snapshot' + '/', '')
            file = file.strip('.log')
            ds_name, method, layer = parse_name(file)
            if ds_name not in datasets[0]:
                continue
            if method == 'intact' and int(layer) == 1:
                continue
            if ds_name not in kernel_map:
                kernel_map[ds_name] = {}
                method_map[ds_name] = {}
            for gk in res:
                gk_method = gk
                if gk.startswith('WL_'):
                    acc_entry = f'WL-{method}-FES'
                    gk_method = 'WL'
                elif gk.startswith('GL_'):
                    acc_entry = f'GL-{method}-FES'
                    gk_method = 'GL'
                elif gk.startswith('WLOA_'):
                    acc_entry = f'WLOA-{method}-FES'
                    gk_method = 'WLOA'
                elif gk.startswith('CSM_'):
                    acc_entry = f'CSM-{method}-FES'
                    gk_method = 'CSM'
                else:
                    acc_entry = f'{gk}-{method}-FES'
                    
                
                if gk_method not in kernel_methods:
                    continue
                if 'acc' not in res[gk]:
                    continue
                if acc_entry not in kernel_map[ds_name]:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
                elif get_diff(res[gk]['acc'], kernel_map[ds_name][acc_entry]) > 0:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
    tmp_map1, tmp_map2 = {}, {}
    for ds_name, ma in kernel_map.items():
        keys = sorted([x for x in ma])
        tmp_map1[ds_name] = {}
        tmp_map2[ds_name] = {}
        for x in keys:
            tmp_map1[ds_name][x] = kernel_map[ds_name][x]
            tmp_map2[ds_name][x] = method_map[ds_name][x]
    kernel_map = tmp_map1
    method_map = tmp_map2

    print('## Graph Kernels (self-defined filtration)')
    print('### acc')
    df = pd.DataFrame(kernel_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys'
    ))
    print('### parameter')
    df = pd.DataFrame(method_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys'
    ))

    '''
    Graph Kernels: attr/vattr filtration
    '''

    kernel_map, method_map = {}, {}

    for filtration in filtration_methods_attr:

        # FEG
        files = glob.glob(osp.join(f'log0-{filtration}', '*.log'))
        for file in files:
            with open(file, 'r') as f:
                res = json.load(f)
            file = file.replace(f'log0-{filtration}' + '/', '')
            file = file.strip('.log')
            ds_name, method, layer = parse_name(file)
            if ds_name not in datasets[1] and ds_name not in datasets[2]:
                continue
            if method == 'intact' and int(layer) == 1:
                continue
            if ds_name not in kernel_map:
                kernel_map[ds_name] = {}
                method_map[ds_name] = {}
            for gk in res:
                gk_method = gk
                if gk.startswith('WL_'):
                    acc_entry = f'WL-{method}-FEG'
                    gk_method = 'WL'
                elif gk.startswith('GL_'):
                    acc_entry = f'GL-{method}-FEG'
                    gk_method = 'GL'
                elif gk.startswith('WLOA_'):
                    acc_entry = f'WLOA-{method}-FEG'
                    gk_method = 'WLOA'
                elif gk.startswith('CSM_'):
                    acc_entry = f'CSM-{method}-FEG'
                    gk_method = 'CSM'
                else:
                    acc_entry = f'{gk}-{method}-FEG'
                    
                
                # if gk_method not in kernel_methods:
                #     continue
                # print(gk)
                if 'acc' not in res[gk]:
                    continue
                if acc_entry not in kernel_map[ds_name]:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
                elif get_diff(res[gk]['acc'], kernel_map[ds_name][acc_entry]) > 0:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
        # FES
        files = glob.glob(osp.join(f'log0-{filtration}-snapshot', '*.log'))
        for file in files:
            with open(file, 'r') as f:
                res = json.load(f)
            file = file.replace(f'log0-{filtration}-snapshot' + '/', '')
            file = file.strip('.log')
            ds_name, method, layer = parse_name(file)
            if ds_name not in datasets[1] and ds_name not in datasets[2]:
                continue
            if method == 'intact' and int(layer) == 1:
                continue
            if ds_name not in kernel_map:
                kernel_map[ds_name] = {}
                method_map[ds_name] = {}
            for gk in res:
                gk_method = gk
                if gk.startswith('WL_'):
                    acc_entry = f'WL-{method}-FES'
                    gk_method = 'WL'
                elif gk.startswith('GL_'):
                    acc_entry = f'GL-{method}-FES'
                    gk_method = 'GL'
                elif gk.startswith('WLOA_'):
                    acc_entry = f'WLOA-{method}-FES'
                    gk_method = 'WLOA'
                elif gk.startswith('CSM_'):
                    acc_entry = f'CSM-{method}-FES'
                    gk_method = 'CSM'
                else:
                    acc_entry = f'{gk}-{method}-FES'
                    
                
                if gk_method not in kernel_methods:
                    continue
                if 'acc' not in res[gk]:
                    continue
                if acc_entry not in kernel_map[ds_name]:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
                elif get_diff(res[gk]['acc'], kernel_map[ds_name][acc_entry]) > 0:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
    tmp_map1, tmp_map2 = {}, {}
    for ds_name, ma in kernel_map.items():
        keys = sorted([x for x in ma])
        tmp_map1[ds_name] = {}
        tmp_map2[ds_name] = {}
        for x in keys:
            tmp_map1[ds_name][x] = kernel_map[ds_name][x]
            tmp_map2[ds_name][x] = method_map[ds_name][x]
    kernel_map = tmp_map1
    method_map = tmp_map2

    print('## Graph Kernels (native attributes)')
    print('### acc')
    df = pd.DataFrame(kernel_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys'
    ))
    print('### parameters')
    df = pd.DataFrame(method_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys'
    ))



    '''
    Graph Kernels: attr/vattr filtration + other kernels
    '''

    kernel_map, method_map = {}, {}

    for filtration in filtration_methods_attr:

        # FEG
        files = glob.glob(osp.join(f'log0-{filtration}-others', '*.log'))
        files.extend(glob.glob(osp.join(f'log0-{filtration}-WLOA', '*.log')))
        for file in files:
            with open(file, 'r') as f:
                res = json.load(f)
            file = file.replace(f'log0-{filtration}-WLOA' + '/', '')
            file = file.replace(f'log0-{filtration}-others' + '/', '')
            file = file.strip('.log')
            ds_name, method, layer = parse_name(file)
            if ds_name not in datasets[1] and ds_name not in datasets[2]:
                continue
            if method == 'intact' and int(layer) == 1:
                continue
            if ds_name not in kernel_map:
                kernel_map[ds_name] = {}
                method_map[ds_name] = {}
            for gk in res:
                gk_method = gk
                if gk.startswith('WL_'):
                    acc_entry = f'WL-{method}-FEG'
                    gk_method = 'WL'
                elif gk.startswith('GL_'):
                    acc_entry = f'GL-{method}-FEG'
                    gk_method = 'GL'
                elif gk.startswith('WLOA_'):
                    acc_entry = f'WLOA-{method}-FEG'
                    gk_method = 'WLOA'
                elif gk.startswith('CSM_'):
                    acc_entry = f'CSM-{method}-FEG'
                    gk_method = 'CSM'
                else:
                    acc_entry = f'{gk}-{method}-FEG'
                    
                
                if gk_method not in kernel_methods:
                    continue
                if 'acc' not in res[gk]:
                    continue
                if acc_entry not in kernel_map[ds_name]:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
                elif get_diff(res[gk]['acc'], kernel_map[ds_name][acc_entry]) > 0:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
        # FES
        files = glob.glob(osp.join(f'log0-{filtration}-snapshot-others', '*.log'))
        files.extend(glob.glob(osp.join(f'log0-{filtration}-snapshot-WLOA', '*.log')))
        for file in files:
            with open(file, 'r') as f:
                res = json.load(f)
            file = file.replace(f'log0-{filtration}-snapshot-WLOA' + '/', '')
            file = file.replace(f'log0-{filtration}-snapshot-others' + '/', '')
            file = file.strip('.log')
            ds_name, method, layer = parse_name(file)
            if ds_name not in datasets[1] and ds_name not in datasets[2]:
                continue
            if method == 'intact' and int(layer) == 1:
                continue
            if ds_name not in kernel_map:
                kernel_map[ds_name] = {}
                method_map[ds_name] = {}
            for gk in res:
                gk_method = gk
                if gk.startswith('WL_'):
                    acc_entry = f'WL-{method}-FES'
                    gk_method = 'WL'
                elif gk.startswith('GL_'):
                    acc_entry = f'GL-{method}-FES'
                    gk_method = 'GL'
                elif gk.startswith('WLOA_'):
                    acc_entry = f'WLOA-{method}-FES'
                    gk_method = 'WLOA'
                elif gk.startswith('CSM_'):
                    acc_entry = f'CSM-{method}-FES'
                    gk_method = 'CSM'
                else:
                    acc_entry = f'{gk}-{method}-FES'
                    
                
                if gk_method not in kernel_methods:
                    continue
                if 'acc' not in res[gk]:
                    continue
                if acc_entry not in kernel_map[ds_name]:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
                elif get_diff(res[gk]['acc'], kernel_map[ds_name][acc_entry]) > 0:
                    kernel_map[ds_name][acc_entry] = parse_acc(res[gk]['acc'])
                    method_map[ds_name][acc_entry] = f'{filtration}-{method}-{int(layer):0>3d}'
    tmp_map1, tmp_map2 = {}, {}
    for ds_name, ma in kernel_map.items():
        keys = sorted([x for x in ma])
        tmp_map1[ds_name] = {}
        tmp_map2[ds_name] = {}
        for x in keys:
            tmp_map1[ds_name][x] = kernel_map[ds_name][x]
            tmp_map2[ds_name][x] = method_map[ds_name][x]
    kernel_map = tmp_map1
    method_map = tmp_map2

    print('## Graph Kernels (native attributes)')
    print('### acc')
    df = pd.DataFrame(kernel_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys'
    ))
    print('### parameters')
    df = pd.DataFrame(method_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys'
    ))


    '''
    GNNs: self-defined filtration
    '''

    def parse_gnn_name(file, ds_entry, method_entry):
        file = file.replace('.json','')
        split_list = file.split('-')
        if len(split_list) == 3:
            split_list.append(0)

        model_name, filtration, lr, dropout = split_list
        split_list = ds_entry.split('-')
        if len(split_list) == 3:
            ds_name, method, snapshot = split_list
        else:
            ds_name1, ds_name2, method, snapshot = split_list
            ds_name = ds_name1 + '-' + ds_name2
        split_list = method_entry.split('-')
        pooling, layer = split_list[:2]
        return model_name, ds_name, filtration, method, lr, dropout, snapshot, layer, pooling

    gnn_map, method_map = {}, {}
    json_files = glob.glob(osp.join(f'*.json'))
    # print(json_files)
    for file in json_files:
        with open(file, 'r') as f:
            res = json.load(f)
        for ds_entry in res:
            for method_entry in res[ds_entry]:
                model_name, ds_name, filtration, method, lr, dropout, snapshot, layer, pooling \
                    = parse_gnn_name(file, ds_entry, method_entry)
                
                accs = np.array(res[ds_entry][method_entry]['acc'])
                if len(accs) < 10:
                    continue
                # print(model_name, ds_name, filtration)
                acc, std = accs.mean() * 100, accs.std() * 100
                if ds_name not in gnn_map:
                    gnn_map[ds_name] = {}
                    method_map[ds_name] = {}
                if method == 'intact' and int(snapshot) == 1:
                    entry = f'{model_name}-original'
                else:
                    entry = f'{model_name}-{method}-FEG'
                
                if entry not in gnn_map[ds_name]:
                    gnn_map[ds_name][entry] = parse_acc(f'{acc:.2f} +- {std:.2f}')
                    method_map[ds_name][entry] = f'({filtration}-{int(snapshot):0>3d})-({pooling}-{layer})-({lr}-{dropout})'
                elif get_diff(f'{acc:.2f} +- {std:.2f}', gnn_map[ds_name][entry]) > 0:
                    gnn_map[ds_name][entry] = parse_acc(f'{acc:.2f} +- {std:.2f}')
                    method_map[ds_name][entry] = f'({filtration}-{int(snapshot):0>3d})-({pooling}-{layer})-({lr}-{dropout})'

    tmp_map1, tmp_map2 = {}, {}
    for ds_name, ma in gnn_map.items():
        # print(ma)
        keys = sorted([x for x in ma])
        tmp_map1[ds_name] = {}
        tmp_map2[ds_name] = {}
        # print(keys)
        for x in keys:
            tmp_map1[ds_name][x] = gnn_map[ds_name][x]
            tmp_map2[ds_name][x] = method_map[ds_name][x]
    gnn_map = tmp_map1
    method_map = tmp_map2

    print('## GNNs')
    print('### acc')
    df = pd.DataFrame(gnn_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys'
    ))
    print('### parameters')
    df = pd.DataFrame(method_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys'
    ))

  
    '''
    GNNs: vary filtration
    '''
    gnn_map, method_map = {}, {}
    json_files = glob.glob(osp.join(f'*.json'))

    for file in json_files:
        with open(file, 'r') as f:
            res = json.load(f)
        for ds_entry in res:
            for method_entry in res[ds_entry]:
                model_name, ds_name, filtration, method, lr, dropout, snapshot, layer, pooling \
                    = parse_gnn_name(file, ds_entry, method_entry)
                
                accs = np.array(res[ds_entry][method_entry]['acc'])
                acc, std = accs.mean() * 100, accs.std() * 100

                if len(accs) < 10:
                    continue
                if method == 'intact' and int(snapshot) == 1:
                    continue
                if model_name not in gnn_map:
                    gnn_map[model_name] = {}
                if filtration not in gnn_map[model_name]:
                    gnn_map[model_name][filtration] = {}
                if ds_name not in gnn_map[model_name][filtration]:
                    gnn_map[model_name][filtration][ds_name] = parse_acc(f'{acc:.2f} +- {std:.2f}')
                if get_diff(f'{acc:.2f} +- {std:.2f}', gnn_map[model_name][filtration][ds_name]) > 0:
                    gnn_map[model_name][filtration][ds_name] = parse_acc(f'{acc:.2f} +- {std:.2f}')

    print('## GNN vary filtration')
    for model_name, ma in gnn_map.items():
        print(f'### {model_name}')
        df = pd.DataFrame(ma)
        print(tabulate.tabulate(
            df,
            tablefmt='github',
            headers='keys'
        ))
import json

import tabulate
import pandas as pd
import numpy as np
import os
import os.path as osp
import glob
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='the log directory')
    args = parser.parse_args()

    path = args.dir

    files = sorted(glob.glob(osp.join(path, '*.log')))
    
    def parse_name(s):
        s = s.replace('-snapshot','')
        l = s.split('-')
        ds_name = ''
        for x in l[:-2]:
            if len(ds_name) > 0:
                ds_name = ds_name + '-'
            ds_name = ds_name + x
        return ds_name, l[-2], l[-1]

    
    def get_diff(x, y):
        if x == 'nan':
            return -1e9
        if y == 'nan':
            return 1e9
        x = x.strip('**')
        y = y.strip('**')
        x_acc, x_std = x.split(' +- ')
        y_acc, y_std = y.split(' +- ')
        return float(x_acc) - float(y_acc)
        if x_acc != y_acc:
            return float(x_acc) - float(y_acc)
        else:
            return float(y_std) - float(x_std)

    acc_map, time_map, diff_map, best_map = {}, {}, {}, {}

    for filename in files:
        with open(filename, 'r') as f:
            filename = filename.replace(path + '/', '')
            filename = filename.strip('.log')

            try:
                json_dict = json.load(f)
            except json.decoder.JSONDecodeError:
                continue
            ds_name, filter_method, snapshot = parse_name(filename)
            for kk, v in json_dict.items():
                k = kk
                if 'WL_' in kk:
                    k = 'WL'
                if 'GL_' in kk:
                    k = 'GL'
                if 'WLOA_' in kk:
                    k = 'WLOA'
                if 'CSM_' in kk:
                    k = 'CSM'
                if 'PM_' in kk:
                    k = 'PM'
                if 'MLG' in kk:
                    k = 'MLG'
                if k not in acc_map:
                    acc_map[k] = {}
                    time_map[k] = {}
                if ds_name not in acc_map[k]:
                    acc_map[k][ds_name] = {}
                    time_map[k][ds_name] = {}
                
                if filter_method + '-' + snapshot not in acc_map[k][ds_name]:
                    acc_map[k][ds_name][filter_method + '-' + snapshot] = 'nan'
                    time_map[k][ds_name][filter_method + '-' + snapshot] = 'nan'
                if 'acc' in v and get_diff(acc_map[k][ds_name][filter_method + '-' + snapshot], v['acc']) < 0:
                    acc_map[k][ds_name][filter_method + '-' + snapshot] = v['acc']
                    time_map[k][ds_name][filter_method + '-' + snapshot] = v['time']
                
    
    for k, ma in acc_map.items():
        print(f'## {k}')
        df = pd.DataFrame(ma)
        print(tabulate.tabulate(
            df,
            tablefmt='github',
            headers='keys'
        ))
        print()
    

    diff_map = {}
    for kernel_method, ma in acc_map.items():
        diff_map[kernel_method] = {}
        for ds_name, mama in ma.items():
            if 'intact-001' not in mama:
                mama['intact-001'] = 'nan'
            intact_1 = mama['intact-001']
            diff_map[kernel_method][ds_name] = -1e9
            for method, res in mama.items():
                if method == 'intact-001':
                    continue
                diff_map[kernel_method][ds_name] = max(get_diff(res, intact_1), diff_map[kernel_method][ds_name])
    
    print(f'## improvement')
    df = pd.DataFrame(diff_map)
    print(tabulate.tabulate(
        df,
        tablefmt='github',
        headers='keys',
        floatfmt='+.2f'
    ))
    print()

    best_map = {}
    for kernel_method, ma in acc_map.items():
        for ds_name, mama in ma.items():
            if ds_name not in best_map:
                best_map[ds_name] = {}
                best_map[ds_name]['name'] = 'nan'
                best_map[ds_name]['acc'] = 'nan'
                best_map[ds_name]['improvement'] = -1e9
            for method, res in mama.items():
                if get_diff(best_map[ds_name]['acc'], res) < 0:
                    best_map[ds_name]['name'] = kernel_method + '-' + method
                    best_map[ds_name]['acc'] = res
    ds_names = set()
    for kernel_method, ma in acc_map.items():
        for ds_name, _ in ma.items():
            ds_names.add(ds_name)
    
    for ds_name in ds_names:
        intact_1 = 'nan'
        for kernel_method, ma in acc_map.items():
            if get_diff(intact_1, ma[ds_name]['intact-001']) < 0:
                intact_1 = ma[ds_name]['intact-001']
        
        for kernel_method, ma in acc_map.items():
            for d_name, mama in ma.items():
                if ds_name != d_name:
                    continue
                for method, res in mama.items():
                    if method == 'intact-001':
                        continue
                    best_map[ds_name]['improvement'] = max(best_map[ds_name]['improvement'], get_diff(res,intact_1))
    
    print('## best method')
    df = pd.DataFrame(best_map)
    print(tabulate.tabulate(
        df.transpose(),
        tablefmt='github',
        headers='keys',
        floatfmt='+.2f'
    ))
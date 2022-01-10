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
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        acc_map = json.load(f)

    res_map = {}
    name_map = {}
    all_map = {}

    def parse_name(s):
        l = s.split('-')
        ds_name = ''
        for x in l[:-2]:
            if len(ds_name) > 0:
                ds_name = ds_name + '-'
            ds_name = ds_name + x
        return ds_name, l[-2], l[-1]

    for k, ma in acc_map.items():
        ds_name, method, snapshot = parse_name(k)
        if ds_name not in res_map:
            res_map[ds_name] = {}
            name_map[ds_name] = {}
            all_map[ds_name] = {}
        if method + '-' + snapshot not in all_map[ds_name]:
            all_map[ds_name][method + '-' + snapshot] = {}
        best_acc, best_std = 0, 0
        for kk, mama in ma.items():
            if isinstance(mama, dict):
                if len(mama['acc']) < 10:
                    continue
                acc = np.array(mama['acc']).mean() * 100
                std = np.array(mama['acc']).std() * 100
            else:
                if len(mama) < 10:
                    continue
                acc = np.array(mama).mean() * 100
                std = np.array(mama).std() * 100
            all_map[ds_name][method + '-' +
                             snapshot][kk] = '{:.2f} +- {:.2f}'.format(acc, std)

            if acc > best_acc:
                best_acc, best_std = acc, std
                best_name = kk
        res_map[ds_name][method + '-' +
                         snapshot] = '{:.2f} +- {:.2f}'.format(best_acc, best_std)
        name_map[ds_name][method + '-' +
                          snapshot] = best_name

    for ds_name, ma in all_map.items():
        print('dataset: {}'.format(ds_name))
        df = pd.DataFrame(ma)
        print(
            tabulate.tabulate(
                df.transpose(),
                tablefmt='github',
                headers='keys',
            )
        )
        print()

    print('\n\n==================================\n\n')

    df = pd.DataFrame(res_map)
    print(
        tabulate.tabulate(
            df,
            tablefmt='github',
            headers='keys',
        )
    )

    print('\n\n==================================\n\n')

    df = pd.DataFrame(name_map)
    print(
        tabulate.tabulate(
            df.transpose(),
            tablefmt='github',
            headers='keys',
        )
    )

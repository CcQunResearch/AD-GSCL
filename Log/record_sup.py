# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 14:38
# @Author  :
# @Email   :
# @File    : record_sup.py
# @Software: PyCharm
# @Note    :
import os
import sys
import json
import math
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '..'))

cal_mean = -10

if __name__ == '__main__':
    log_dir_path = os.path.join(dirname, '..', 'Log')

    for filename in os.listdir(log_dir_path):
        if filename[-4:] == 'json':
            print(f'【{filename[:-5]}】')
            filepath = os.path.join(log_dir_path, filename)

            log = json.load(open(filepath, 'r', encoding='utf-8'))
            print('dataset:', log['dataset'])
            print('tokenize mode:', log['tokenize mode'])
            print('unsup train size:', log['unsup train size'])
            print('split:', log['split'])
            print('batch size:', log['batch size'])
            print('undirected:', log['undirected'])
            if 'model' in log.keys():
                print('model:', log['model'])
            print('n layers feat:', log['n layers feat'])
            print('n layers conv:', log['n layers conv'])
            print('n layers fc:', log['n layers fc'])
            print('vector size:', log['vector size'])
            print('hidden size:', log['hidden'])
            print('global pool:', log['global pool'])
            print('skip connection:', log['skip connection'])
            print('res branch:', log['res branch'])
            print('dropout:', log['dropout'])
            print('edge norm:', log['edge norm'])
            print('t:', log['t'])
            print('lr:', log['lr'])
            print('epochs:', log['epochs'])
            print('weight decay:', log['weight decay'])
            print('gamma:', log['gamma'])
            print('use unlabel:', log['use unlabel'])
            print('use unsup loss:', log['use unsup loss'])
            print('k:', log['k'])

            acc_list = []
            max_epoch_acc_list = []
            for run in log['record']:
                # mean_acc = run['mean acc']
                mean_acc = round(np.mean(run['test accs'][cal_mean:]), 3)
                max_epoch_acc = round(np.max(run['test accs']), 3)
                acc_list.append(mean_acc)
                max_epoch_acc_list.append(max_epoch_acc)

            mean = round(sum(acc_list) / len(acc_list), 3)
            mean_max_epoch = round(sum(max_epoch_acc_list) / len(max_epoch_acc_list), 3)
            sd = round(math.sqrt(sum([(x - mean) ** 2 for x in acc_list]) / len(acc_list)), 3)
            sd_max_epoch = round(
                math.sqrt(sum([(x - mean_max_epoch) ** 2 for x in max_epoch_acc_list]) / len(max_epoch_acc_list)), 3)
            maxx = max(acc_list)
            maxx_max_epoch = max(max_epoch_acc_list)
            print('test acc | max acc: {:.3f}±{:.3f} | {:.3f}'.format(mean, sd, maxx))
            print('test acc | max acc (max epoch): {:.3f}±{:.3f} | {:.3f}'.format(mean_max_epoch, sd_max_epoch,
                                                                                  maxx_max_epoch))
            print()
            print()


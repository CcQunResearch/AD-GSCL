# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:56
# @Author  :
# @Email   :
# @File    : main(pretrain).py
# @Software: PyCharm
# @Note    :
import sys
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
from Main.pargs import pargs
from Main.dataset import TreeDataset
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.sort import sort_dataset
from Main.model import ResGCN_graphcl, BiGCN_graphcl, InfoMax_graphcl
from Main.utils import create_log_dict_pretrain, write_log, write_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def pre_train(dataloader, model, optimizer, device):
    model.train()
    total_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)

        loss = model.infomax_loss(data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(dataloader.dataset)


def semisup_train(unsup_train_loader, train_loader, nrlabel, model, optimizer, device, gamma):
    model.train()
    total_loss = 0

    for sup_data, unsup_data in zip(train_loader, unsup_train_loader):
        optimizer.zero_grad()
        sup_data = sup_data.to(device)
        unsup_data = unsup_data.to(device)

        sup_out = model(sup_data)
        sup_loss = F.nll_loss(sup_out, sup_data.y.view(-1))

        sup_data_list = sup_data.to_data_list()
        unsup_data_list = unsup_data.to_data_list()
        for item in unsup_data_list:
            item.y = torch.LongTensor([nrlabel]).to(device)
        data = Batch.from_data_list(sup_data_list + unsup_data_list).to(device)

        out = model.forward_graphrep(data)
        cl_loss = model.contrastive_loss(out, data.y, device)

        loss = sup_loss + gamma * cl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sup_data.num_graphs

    return total_loss / len(train_loader.dataset)


def test(model, dataloader, num_classes, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        error += F.nll_loss(pred, data.y.long().view(-1)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.max(1).indices.tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = round(accuracy_score(y_true, y_pred), 4)
    precs = []
    recs = []
    f1s = []
    for label in range(num_classes):
        precs.append(round(precision_score(y_true == label, y_pred == label, labels=True), 4))
        recs.append(round(recall_score(y_true == label, y_pred == label, labels=True), 4))
        f1s.append(round(f1_score(y_true == label, y_pred == label, labels=True), 4))
    micro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)

    macro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    return error / len(dataloader.dataset), acc, precs, recs, f1s, \
           [micro_p, micro_r, micro_f1], [macro_p, macro_r, macro_f1]


def test_and_log(model, val_loader, test_loader, num_classes, device, epoch, lr, loss, train_acc, ft_log_record):
    val_error, val_acc, val_precs, val_recs, val_f1s, val_micro_metric, val_macro_metric = \
        test(model, val_loader, num_classes, device)
    test_error, test_acc, test_precs, test_recs, test_f1s, test_micro_metric, test_macro_metric = \
        test(model, test_loader, num_classes, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Val ERROR: {:.7f}, Test ERROR: {:.7f}\n  Train ACC: {:.4f}, Validation ACC: {:.4f}, Test ACC: {:.4f}\n' \
                   .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc) \
               + f'  Test PREC: {test_precs}, Test REC: {test_recs}, Test F1: {test_f1s}\n' \
               + f'  Test Micro Metric(PREC, REC, F1):{test_micro_metric}, Test Macro Metric(PREC, REC, F1):{test_macro_metric}'

    ft_log_record['val accs'].append(val_acc)
    ft_log_record['test accs'].append(test_acc)
    ft_log_record['test precs'].append(test_precs)
    ft_log_record['test recs'].append(test_recs)
    ft_log_record['test f1s'].append(test_f1s)
    ft_log_record['test micro metric'].append(test_micro_metric)
    ft_log_record['test macro metric'].append(test_macro_metric)
    return val_error, log_info, ft_log_record


if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    ft_runs = args.ft_runs

    word_embedding = 'tfidf' if 'tfidf' in dataset else 'word2vec'
    lang = 'ch' if 'Weibo' in dataset else 'en'
    tokenize_mode = args.tokenize_mode

    split = args.split
    batch_size = args.batch_size
    undirected = args.undirected

    weight_decay = args.weight_decay
    epochs = args.epochs
    ft_epochs = args.ft_epochs
    gamma = args.gamma

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', unsup_dataset, 'dataset')
    model_path = osp.join(dirname, '..', 'Model',
                          f'w2v_{dataset}_{tokenize_mode}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')
    weight_path = osp.join(dirname, '..', 'Model', f'{log_name}.pt')

    log = open(log_path, 'w')
    log_dict = create_log_dict_pretrain(args)

    if not osp.exists(model_path) and word_embedding == 'word2vec':
        sentences = collect_sentences(label_source_path, unlabel_dataset_path, unsup_train_size, lang, tokenize_mode)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    word2vec = Embedding(model_path, lang, tokenize_mode) if word_embedding == 'word2vec' else None

    for run in range(runs):
        unlabel_dataset = TreeDataset(unlabel_dataset_path, word_embedding, word2vec, undirected)
        unsup_train_loader = DataLoader(unlabel_dataset, batch_size, shuffle=True)
        # unsup_train_loader = DataLoader(unlabel_dataset, 8, shuffle=True)

        num_classes = 4 if 'Twitter' in dataset or dataset == 'PHEME' else 2
        nrlabel = 0 if num_classes == 2 else 3
        if args.model == 'ResGCN':
            model = ResGCN_graphcl(dataset=unlabel_dataset, t=args.t, num_classes=num_classes, hidden=args.hidden,
                                   num_feat_layers=args.n_layers_feat, num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc, gfn=False, collapse=False,
                                   residual=args.skip_connection, res_branch=args.res_branch,
                                   global_pool=args.global_pool, dropout=args.dropout,
                                   edge_norm=args.edge_norm).to(device)
        elif args.model == 'BiGCN':
            model = BiGCN_graphcl(unlabel_dataset.num_features, args.hidden, args.hidden, args.t, num_classes).to(
                device)
        elif args.model == 'GIN':
            model = InfoMax_graphcl(unlabel_dataset.num_features, args.hidden, args.t, num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

        write_log(log, f'runs:{run}')
        log_record = {
            'run': run,
            'record': []
        }

        for epoch in range(1, epochs + 1):
            pretrain_loss = pre_train(unsup_train_loader, model, optimizer, device)

            log_info = 'Epoch: {:03d}, Loss: {:.7f}'.format(epoch, pretrain_loss)
            write_log(log, log_info)

        torch.save(model.state_dict(), weight_path)
        write_log(log, '')

        # ks = [10, 20, 40, 80, 100, 200, 300, 500, 10000]
        ks = [10000]
        for k in ks:
            for r in range(ft_runs):
                ft_lr = args.ft_lr
                write_log(log, f'k:{k}, r:{r}')

                ft_log_record = {'k': k, 'r': r, 'val accs': [], 'test accs': [], 'test precs': [], 'test recs': [],
                                 'test f1s': [], 'test micro metric': [], 'test macro metric': []}

                sort_dataset(label_source_path, label_dataset_path, k_shot=k, split=split)

                train_dataset = TreeDataset(train_path, word_embedding, word2vec, undirected)
                val_dataset = TreeDataset(val_path, word_embedding, word2vec, undirected)
                test_dataset = TreeDataset(test_path, word_embedding, word2vec, undirected)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                model.load_state_dict(torch.load(weight_path))
                optimizer = Adam(model.parameters(), lr=args.ft_lr, weight_decay=weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

                val_error, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, num_classes,
                                                                  device, 0, args.ft_lr, 0, 0, ft_log_record)
                write_log(log, log_info)

                for epoch in range(1, ft_epochs + 1):
                    ft_lr = scheduler.optimizer.param_groups[0]['lr']
                    _ = semisup_train(unsup_train_loader, train_loader, nrlabel, model, optimizer, device, gamma)

                    train_error, train_acc, _, _, _, _, _ = test(model, train_loader, num_classes, device)
                    val_error, log_info, ft_log_record = test_and_log(model, val_loader, test_loader, num_classes,
                                                                      device, epoch, ft_lr, train_error, train_acc,
                                                                      ft_log_record)
                    write_log(log, log_info)

                    if split == '622':
                        scheduler.step(val_error)

                ft_log_record['mean acc'] = round(np.mean(ft_log_record['test accs'][-10:]), 3)
                log_record['record'].append(ft_log_record)
                write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)

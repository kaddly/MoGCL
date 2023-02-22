import os
import random
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import dataset, dataloader


def read_YelpChi(path=os.path.join(os.path.abspath('.'), 'data'), file_name='YelpChi.mat'):
    data_dir = os.path.join(path, file_name)
    yelp = sio.loadmat(data_dir)
    net_rur = yelp['net_rur']
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    feat_data = yelp['features'].todense.A
    labels = yelp['label'].flatten()
    return [net_rur, net_rtr, net_rsr], feat_data, labels


def read_Amazon(path=os.path.join(os.path.abspath('.'), 'data'), file_name='Amazon.mat'):
    data_dir = os.path.join(path, file_name)
    amz = sio.loadmat(data_dir)
    net_upu = amz['net_upu']
    net_usu = amz['net_usu']
    net_uvu = amz['net_uvu']
    feat_data = amz['features'].todense.A
    labels = amz['label'].flatten()
    return [net_upu, net_usu, net_uvu], feat_data, labels


def read_data(data_set):
    if data_set == 'amz':
        return read_Amazon()
    elif data_set == 'yelp':
        return read_YelpChi()


def generator_pos_neg_nodes(nodes, relation_list):
    pass


def generator_neighbors(nodes, relation):
    pass


def load_data(args):
    relation_list, feat_data, labels = read_data(args.dataset)
    # train_test_split
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.dataset == 'yelp':
        index = list(range(len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=args.test_size, random_state=2, shuffle=True)
    elif args.dataset == 'amz':
        index = list(range(3305, len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:], test_size=args.test_size, random_state=2, shuffle=True)



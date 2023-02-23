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
    all_pos_neg_nodes = []
    relation_homo = None
    for relation in relation_list:
        if relation_homo is None:
            relation_homo = relation
        relation_homo += relation
    relation_homo = relation_homo.A
    for node in nodes:
        all_pos_neg_nodes.append(relation_homo[node])
    return all_pos_neg_nodes


def generator_neighbors(nodes, relation_list):
    all_neighbors = [[[] for _ in range(len(relation_list))] for _ in range(len(nodes))]
    for i, relation in enumerate(relation_list):
        for node in nodes:
            neigh = relation[node].nonzero()[1]
            if not neigh:
                neigh = [node]
            all_neighbors[node][i].append(neigh)
    return all_neighbors


class MultiViewDataset(dataset):
    def __init__(self, nodes):
        super(MultiViewDataset, self).__init__()
        self.nodes = nodes

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, item):
        return self.nodes[item]


class Collect_fn:
    def __init__(self, all_neighbors, all_pos_neg_nodes, num_neigh, num_pos, num_view):
        self.all_neighbors = all_neighbors
        self.all_pos_neg_nodes = all_pos_neg_nodes
        self.nun_neigh = num_neigh
        self.num_pos = num_pos
        self.num_view = num_view

    def __call__(self, data):
        nodes, nodes_neigh, nodes_pos, nodes_pos_neigh = [], [], [], []
        for node in data:
            pass


def load_data(args):
    relation_list, feat_data, labels = read_data(args.dataset)
    # train_test_split
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.dataset == 'yelp':
        index = list(range(len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
                                                                test_size=args.test_size, random_state=2, shuffle=True)
    elif args.dataset == 'amz':
        index = list(range(3305, len(labels)))
        idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=args.test_size, random_state=2, shuffle=True)
    all_neighbors = generator_neighbors(list(range(len(labels))), relation_list)
    all_pos_neg_nodes = generator_pos_neg_nodes(list(range(len(labels))), relation_list)

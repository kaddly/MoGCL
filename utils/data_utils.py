import os
import random
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


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


def generator_pos_neg_nodes(nodes, relation_list, num_pos):
    nodes_pos, nodes_neg = [], []
    relation_homo = None
    for relation in relation_list:
        if relation_homo is None:
            relation_homo = relation
        relation_homo += relation
    for node in nodes:
        keys = relation_homo[node].data
        if len(keys) < num_pos:
            pos = relation_homo[node].nonzero()[1].tolist()
            nodes_pos.append(pos + [node for _ in (num_pos-len(keys))])
            nodes_neg.append(np.delete(nodes, pos+node).tolist()[:len(nodes)-num_pos])
        else:
            pos_neg = sorted(relation_homo[node].nonzero()[1].tolist(), key=keys)
            nodes_pos.append(pos_neg[:num_pos])
            nodes_neg.append(np.delete(nodes, nodes_pos[node]).tolist())
    return nodes_pos, nodes_neg


def generator_neighbors(nodes, relation_list):
    all_neighbors = [[[] for _ in range(len(relation_list))] for _ in range(len(nodes))]
    for i, relation in enumerate(relation_list):
        for node in nodes:
            neigh = relation[node].nonzero()[1]
            if not neigh:
                neigh = [node]
            all_neighbors[node][i].append(neigh)
    return all_neighbors


class MultiViewDataset(Dataset):
    def __init__(self, nodes):
        super(MultiViewDataset, self).__init__()
        self.nodes = nodes

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, item):
        return self.nodes[item]


class Collate_fn:
    def __init__(self, all_neighbors, all_pos, all_neg, num_neigh):
        self.all_neighbors = all_neighbors
        self.num_neigh = num_neigh
        self.all_pos = all_pos
        self.all_neg = all_neg

    def __call__(self, data):
        nodes, nodes_neigh, nodes_pos, nodes_pos_neigh, nodes_neg = [], [], [], [], []
        for node in data:
            nodes.append(node)
            nodes_pos.append(self.all_pos[node])
            nodes_neg.append(self.all_neg[node])
            neigh = []
            for candidates in self.all_neighbors[node]:
                neigh.append(random.choices(candidates, k=self.num_neigh))
            nodes_neigh.append(neigh)
            node_pos_neigh = []
            for pos in nodes_pos[node]:
                pos_neigh = []
                for candidates in self.all_neighbors[pos]:
                    pos_neigh.append(random.choices(candidates, k=self.num_neigh))
                node_pos_neigh.append(pos_neigh)
            nodes_pos_neigh.append(node_pos_neigh)
        return (torch.tensor(nodes), torch.tensor(nodes_neigh)), (torch.tensor(nodes_pos), torch.tensor(nodes_pos_neigh)), torch.tensor(nodes_neg)


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
    all_pos, all_neg = generator_pos_neg_nodes(list(range(len(labels))), relation_list, args.num_pos)
    train_dataset = MultiViewDataset(idx_train)
    collate_fn = Collate_fn(all_neighbors, all_pos, all_neg, args.num_neigh)
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size,collate_fn=collate_fn)
    return train_iter, feat_data, (idx_test, y_test)

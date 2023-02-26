import os
import random
import numpy as np
import scipy.io as sio
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def read_YelpChi(path=os.path.join(os.path.abspath('.'), 'data'), file_name='YelpChi.mat'):
    data_dir = os.path.join(path, file_name)
    yelp = sio.loadmat(data_dir)
    net_rur = yelp['net_rur']
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    feat_data = yelp['features'].todense().A
    labels = yelp['label'].flatten()
    return [net_rur, net_rtr, net_rsr], feat_data, labels


def read_Amazon(path=os.path.join(os.path.abspath('.'), 'data'), file_name='Amazon.mat'):
    data_dir = os.path.join(path, file_name)
    amz = sio.loadmat(data_dir)
    net_upu = amz['net_upu']
    net_usu = amz['net_usu']
    net_uvu = amz['net_uvu']
    feat_data = amz['features'].todense().A
    labels = amz['label'].flatten()
    return [net_upu, net_usu, net_uvu], feat_data, labels


def read_data(data_set):
    if data_set == 'amz':
        return read_Amazon()
    elif data_set == 'yelp':
        return read_YelpChi()


def generator_pos_neg_nodes(nodes, relation_list, args):
    print("loading pos_neg_nodes")
    if os.path.isfile(os.path.join('data', args.dataset + '_pos_neg.pkl')):
        f_read = open(os.path.join('data', args.dataset + '_pos_neg.pkl'), 'rb')
        nodes_pos, nodes_neg = pickle.load(f_read)
        f_read.close()
        return nodes_pos, nodes_neg
    num_pos = args.num_pos
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
            nodes_pos.append(pos + [node for _ in range(num_pos - len(keys))])
            nodes_neg.append(np.delete(nodes, pos + [node]).tolist()[:len(nodes) - num_pos])
        else:
            pos_neg = relation_homo[node].nonzero()[1][np.argsort(keys)]
            nodes_pos.append(pos_neg[:num_pos].tolist())
            nodes_neg.append(np.delete(nodes, nodes_pos[-1]).tolist())
    f_save = open(os.path.join('data', args.dataset + '_pos_neg.pkl'), 'wb')
    pickle.dump([nodes_pos, nodes_neg], f_save)
    f_save.close()
    return nodes_pos, nodes_neg


def generator_neighbors(nodes, relation_list, args):
    print("loading neighbors")
    if os.path.isfile(os.path.join('data', args.dataset + '_neighbors.pkl')):
        f_read = open(os.path.join('data', args.dataset + '_neighbors.pkl'), 'rb')
        all_neighbors = pickle.load(f_read)
        f_read.close()
        return all_neighbors
    all_neighbors = [[[] for _ in range(len(relation_list))] for _ in range(len(nodes))]
    for i, relation in enumerate(relation_list):
        for node in nodes:
            neigh = relation[node].nonzero()[1].tolist()
            if not neigh:
                neigh = [node]
            all_neighbors[node][i].extend(random.choices(neigh, k=args.num_neigh))
    f_save = open(os.path.join('data', args.dataset + '_neighbors.pkl'), 'wb')
    pickle.dump(all_neighbors, f_save)
    f_save.close()
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
    def __init__(self, all_neighbors, all_pos, all_neg):
        assert len(all_neighbors) == len(all_pos) == len(all_neg)
        self.all_neighbors = all_neighbors
        self.all_pos = all_pos
        self.all_neg = all_neg

    def get_nodes_neigh(self, nodes):
        nodes_neigh = []
        for node in nodes:
            nodes_neigh.append(self.all_neighbors[node])
        return [nodes, nodes_neigh]

    def __call__(self, data):
        nodes, nodes_neigh, nodes_pos, nodes_pos_neigh, nodes_neg = [], [], [], [], []
        for node in data:
            nodes.append(node)
            nodes_pos.append(self.all_pos[node])
            nodes_neg.append(self.all_neg[node])
            nodes_neigh.append(self.all_neighbors[node])
            pos_neigh = []
            for pos in nodes_pos[-1]:
                pos_neigh.append(self.all_neighbors[pos])
            nodes_pos_neigh.append(pos_neigh)
        return [torch.tensor(nodes), torch.tensor(nodes_neigh), torch.tensor(nodes_pos), torch.tensor(nodes_pos_neigh),
                torch.tensor(nodes_neg)]


def load_data(args):
    relation_list, feat_data, labels = read_data(args.dataset)
    # train_test_split
    if args.dataset == 'yelp':
        index = list(range(len(labels)))
        idx_train, idx_val, y_train, y_val = train_test_split(index, labels, stratify=labels, test_size=args.test_size,
                                                              random_state=args.seed, shuffle=True)
    elif args.dataset == 'amz':
        index = list(range(3305, len(labels)))
        labels = labels[3305:]
        idx_train, idx_val, y_train, y_val = train_test_split(index, labels, stratify=labels, test_size=args.test_size,
                                                              random_state=args.seed, shuffle=True)
    else:
        raise ValueError("unsupported dataset")
    all_neighbors = generator_neighbors(list(range(feat_data.shape[0])), relation_list, args)
    all_pos, all_neg = generator_pos_neg_nodes(list(range(feat_data.shape[0])), relation_list, args)
    train_dataset = MultiViewDataset(idx_train)
    collate_fn = Collate_fn(all_neighbors, all_pos, all_neg)
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, collate_fn=collate_fn)
    return train_iter, feat_data, collate_fn(idx_val), collate_fn.get_nodes_neigh(index), (idx_train, idx_val, y_train, y_val)


def setup_logging(dataset):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("embeds", exist_ok=True)
    os.makedirs(os.path.join("models", dataset), exist_ok=True)
    os.makedirs(os.path.join("results", dataset), exist_ok=True)
    os.makedirs(os.path.join("embeds", dataset), exist_ok=True)

import numpy as np
import random
import torch
from utils import set_params
from train import train, evaluate
from utils import load_data, setup_logging
from module import MoGCL, LogReg

args = set_params()
np.random.seed(args.seed)
random.seed(args.seed)


def set_train_params():
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.batch_size = 128
    args.epochs = 100
    args.start_epoch = 0

    args.lr = 0.0008
    args.weight_decay = 0
    args.momentum = 0.9

    args.resume = ""
    args.arch = 'MoGCL' + str(args.dim)
    args.patience = 10
    args.print_freq = 10

    args.eva_lr = 0.01
    args.eva_wd = 0


def set_model_params():
    args.dim = 64
    args.attn_size = 64
    args.feat_drop = -1
    args.attn_drop = 0.3
    args.moco_m = 0.998
    args.moco_t = 0.07
    args.is_mlp = True


def launch():
    set_model_params()
    set_train_params()
    train(args)
    # test_attention()


def test_attention():
    # load data
    _, feat_data, _, _, _ = load_data(args)
    device = args.device
    model = MoGCL(feat_data, args.dim, args.num_view, args.num_pos, args.num_neigh,
                  args.attn_size, args.feat_drop, args.attn_drop, len(feat_data), args.moco_m, args.moco_t, args.is_mlp)
    checkpoint = torch.load("./models/amz/model_best.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    node = torch.tensor([11762])
    node_neighbors = torch.tensor([[[5959, 359, 3328, 5771, 8849, 8260, 5654, 9126, 9126, 5654],
                                   [1863, 7628, 5034, 5233, 8694, 4244, 4244, 5233, 4244, 1360],
                                   [4667, 7386, 11190, 179, 179, 7356, 7718, 4667, 179, 7070]]])
    model.get_embeds(node, node_neighbors)


if __name__ == '__main__':
    launch()

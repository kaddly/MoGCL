import numpy as np
import random
import torch
from utils import set_params
from train import train, evaluate

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
    args.arch = 'MoGCL'+str(args.dim)
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


if __name__ == '__main__':
    launch()

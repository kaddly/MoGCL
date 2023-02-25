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
    args.epochs = 500
    args.start_epoch = 0

    args.lr = 0.008
    args.weight_decay = 0

    args.resume = ""
    args.arch = 'MoGCL'+str(args.dim)
    args.patience = 40
    args.print_freq = 50

    args.eva_lr = 0.01
    args.eva_wd = 0


def set_model_params():
    args.dim = 64
    args.attn_size = 64
    args.feat_drop = 0.8
    args.attn_drop = 0.8
    args.mco_m = 0.99
    args.mco_t = 0.5
    args.is_mlp = True


def launch():
    set_model_params()
    set_train_params()
    train(args)


if __name__ == '__main__':
    launch()

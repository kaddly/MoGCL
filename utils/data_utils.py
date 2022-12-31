import os
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
from torch.utils.data import dataset, dataloader


def read_data(path=os.path.join(os.path.abspath('../'), 'data'), file_name='Amazon.mat'):
    data_dir = os.path.join(path, file_name)
    return sio.loadmat(data_dir)


mat_file = read_data()
print(mat_file)

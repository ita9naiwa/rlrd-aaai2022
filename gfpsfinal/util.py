import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils import shuffle


def get_util_range(num_proc):
    util = [str(x) for x in range(10, num_proc * 100, 10)]
    ret = []
    for x in util:
        if len(x) == 2:
            ret.append('0.' + x)
        else:
            ret.append(x[:len(x) - 2] + '.' + x[len(x) - 2:])

    return ret


def input_transform(inputs, use_deadline=False):
    """
    Args:¡
        inputs: [batch_size, seq_len, 3]
    """

    if use_deadline is False:      # implicit
        print("IMPLICIT")
        inputs[:, :, 2] = inputs[:, :, 0]

    inputs = inputs.float()

    div = torch.stack([
        (inputs[:, :, 1] / inputs[:, :, 0]), # Utilization
        (inputs[:, :, 2] / inputs[:, :, 0]),
        (inputs[:, :, 0] - inputs[:, :, 1]) / 1000,
        (inputs[:, :, 0] - inputs[:, :, 2]) / 1000,
        torch.log(inputs[:, :, 0]) / np.log(2), # Log
        torch.log(inputs[:, :, 1] / np.log(2)),
        torch.log(inputs[:, :, 2] / np.log(2)),
        ], dim=-1)

    tt = (inputs / 1000)
    ret = torch.cat([div, tt], dim=-1).type(torch.FloatTensor)

    return ret


class Datasets(Dataset):
    def __init__(self, l):
        super(Datasets, self).__init__()
        ret = []
        le = []
        try:
            for dd in l:
                ret.append(dd.data_set)
        except AttributeError:
            for dd in l:
                ret.append(dd)

        self.data_set = np.vstack(ret)

    def setlen(self, newlen):
        self.data_set = shuffle(self.data_set)
        self.data_set = self.data_set[:newlen]

    def __len__(self):
        return self.data_set.shape[0]

    def __getitem__(self, idx):
        return idx, self.data_set[idx]


def load_datasets(num_procs, num_tasks):
    util_range = get_util_range(num_procs)
    if num_procs == 2:
        range_l = "1.00"
        range_r = "1.80"
    elif num_procs == 4:
        range_l = "2.40"
        range_r = "3.60"
    elif num_procs == 6:
        range_l = "3.60"
        range_r = "5.60"
    elif num_procs == 8:
        range_l = "5.00"
        range_r = "7.40"
    else:
        raise LookupError("지원하지 않는 프로세스 갯수")
    trsets = []
    tesets = []
    on = False

    for util in util_range:
        if util == range_l:
            on = True
        if on:
            with open("../gfpsdata/tr/%d-%d/%s" % (num_procs, num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                trsets.append(ts)
            with open("../gfpsdata/te/%d-%d/%s" % (num_procs, num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                tesets.append(ts)
        if util == range_r:
            break
    return trsets, tesets
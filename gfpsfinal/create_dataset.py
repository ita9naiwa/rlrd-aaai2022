import os
from collections import namedtuple as tup
import argparse


import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sched import SchedT1Dataset
from sched_solver import Solver
from sched_heuristic import liu_test
import sched_heuristic as heu
import pickle

Args = tup("Argument", "seq_len num_procs num_sample period_range util")
NUM_SAMPLES = 200000        # Do not change this

import sched
import importlib
importlib.reload(sched)

NUM_PROC_LIST = [4, 8]
NUM_TASKS_LIST = [32, 64]


def gen_taskset(num_procs, seq_len, ddir="data"):
    try:
        os.mkdir(ddir)
    except FileExistsError:
        pass
    base = "%s/%d-%d" % (ddir, num_procs, seq_len)
    try:
        os.mkdir(base)
    except FileExistsError:
        pass


    step = 0.1

    for util in np.arange(step, num_procs, step):
        args = Args(seq_len=seq_len,
            num_procs=num_procs,
            num_sample=NUM_SAMPLES,
            period_range=(10, 10000),
            util=util)
        dataset = sched.SchedT1Dataset(args.num_procs, args.seq_len, args.num_sample, args.period_range, args.util, gg=True)
        with open(os.path.join(base, "%0.2f" % util), 'wb') as f:
            pickle.dump(dataset, f)


samples = []
for num_proc in NUM_PROC_LIST:
    for num_tasks in NUM_TASKS_LIST:
        if num_proc >= num_tasks:
            continue
        samples.append((num_proc, num_tasks))


from concurrent.futures import ProcessPoolExecutor
E = ProcessPoolExecutor(32)

trdirectory = os.path.dirname("../gfpsdata/tr")
tedirectory = os.path.dirname("../gfpsdata/te")
valdirectory = os.path.dirname("../gfpsdata/val")
if not os.path.exists(trdirectory):
    os.makedirs(trdirectory)
if not os.path.exists(tedirectory):
    os.makedirs(tedirectory)
if not os.path.exists(valdirectory):
    os.makedirs(valdirectory)

def wrap(x):
    print("!")
    num_proc, num_task = x
    print(num_proc, num_task)
    gen_taskset(num_proc, num_task, "../gfpsdata/tr")
    gen_taskset(num_proc, num_task, "../gfpsdata/te")
    gen_taskset(num_proc, num_task, "../gfpsdata/val")
    return "!"

for ret in E.map(wrap, samples):
    print(ret)

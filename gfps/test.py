from functools import wraps
from concurrent.futures import ProcessPoolExecutor
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import shuffle
from sched_solver import Solver
import cy_heuristics as heu
from linearsolver import LinearSolver
from linearsolver import sample_gumbel, get_rank
import time

test_module = heu.test_RTA_LC
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=32)
parser.add_argument("--num_procs", type=int, default=4)
parser.add_argument("--num_test_dataset", type=int, default=100)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--use_deadline", action="store_true")
parser.add_argument("--range_l", type=str, default="3.00")
parser.add_argument("--range_r", type=str, default="3.00")
parser.add_argument("--use_cuda", action="store_true")

confidence = 0.05

args = parser.parse_args()
use_deadline = args.use_deadline


def wrap(x):
    _sample, num_proc, use_deadline = x
    return heu.OPA(_sample, num_proc, None, use_deadline)


def dm_wrap(x):
    _sample, num_proc, use_deadline = x
    return heu.test_RTA_LC(_sample, num_proc, 1, use_deadline)


def timer(func):
    @wraps(func)
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        print("Function {name}, Time : {time:.3f} with result {result}"
              .format(name=func.__name__, time=end-start, result=result))
        return result
    return wrapper


def get_util_range(num_proc):
    util = [str(x) for x in range(10, num_proc * 100, 10)]
    ret = []
    for x in util:
        if len(x) == 2:
            ret.append('0.' + x)
        else:
            ret.append(x[:len(x) - 2] + '.' + x[len(x) - 2:])
    return ret


class Datasets(Dataset):
    def __init__(self, l):
        super(Datasets, self).__init__()
        ret = []
        for dd in l:
            ret.append(dd.data_set)
        self.data_set = np.vstack(ret)

    def setlen(self, newlen):
        self.data_set = shuffle(self.data_set)
        self.data_set = self.data_set[:newlen]

    def __len__(self):
        return self.data_set.shape[0]

    def __getitem__(self, idx):
        return idx, self.data_set[idx]


@timer
def test_heu(eval_dataset, mode="OPA"):
    with ProcessPoolExecutor(max_workers=1) as executor:
        inputs = []
        res_opa = np.zeros(len(eval_dataset), dtype=int).tolist()
        for i, sample in eval_dataset:
            inputs.append((sample, args.num_procs, use_deadline))
        for i, ret in tqdm(enumerate(executor.map(wrap, inputs))):
            res_opa[i] = ret
        opares = np.sum(res_opa)
    return opares


@timer
def test_dm(eval_dataset):
    # with ProcessPoolExecutor(max_workers=1) as executor:
    inputs = []
    res_dm = np.zeros(len(eval_dataset), dtype=int).tolist()
    for i, sample in eval_dataset:
        inputs.append((sample, args.num_procs, use_deadline))
        # print(sample)
    # print(inputs[0])
    dm_wrap(inputs[0])
    # print("run")
    # for i, ret in tqdm(enumerate(executor.map(dm_wrap, inputs))):
    #     res_dm[i] = ret
    # operas = np.sum(res_dm)
    # return operas


@timer
def test_gumbel(model, eval_loader, gumbel_number):
    val = 0
    for i, batch in eval_loader:
        with torch.no_grad():
            linear_score = model(batch, normalize=True)
        gumbel_score = sample_gumbel(linear_score, sampling_number=gumbel_number)
        # gumbel_score : [batch_size x num_gumbel_sample x num_tasks]
        gumbel_rank = get_rank(gumbel_score)  # [batch_size x num_gumbel_sample x num_tasks]
        for j, order in enumerate(gumbel_rank):  # j : ~batch size
            for k, orderd in enumerate(gumbel_rank[j]):  # k : ~ num_gumbel_sample
                x = test_module(batch[j].numpy(), args.num_procs, orderd, False, False)
                if x == 1:
                    val += 1
                    break
                else:
                    continue
    return val


@timer
def test_global_reinforce(model, eval_loader):
    ret = []
    for i, batch in eval_loader:
        with torch.no_grad():
            _, _, actions = model(batch, argmax=True)
        for j, chosen in enumerate(actions.cpu().numpy()):
            order = np.zeros_like(chosen)
            for p in range(args.num_tasks):
                order[chosen[p]] = args.num_tasks - p - 1
            ret.append(test_module(batch[j].numpy(), args.num_procs, order, use_deadline, False))
    return sum(ret)


@timer
def test_reinforce(model, eval_loader):
    ret = []
    for i, batch in eval_loader:
        with torch.no_grad():
            _, _, actions = model(batch, argmax=True)
        for j, chosen in enumerate(actions.cpu().numpy()):
            order = np.zeros_like(chosen)
            for p in range(args.num_tasks):
                order[chosen[p]] = args.num_tasks - p - 1
            ret.append(test_module(batch[j].numpy(), args.num_procs, order, use_deadline, False))
    return sum(ret)


@timer
def test_distillation(model, eval_loader):
    ret = []
    for i, batch in eval_loader:
        with torch.no_grad():
            score = model(batch).detach().numpy()
        argsort = np.argsort(-score)
        for j, chosen in enumerate(argsort):
            order = np.zeros_like(chosen).squeeze()
            for p in range(args.num_tasks):
                order[chosen[p]] = args.num_tasks - p - 1
            ret.append(test_module(batch[j].numpy(), args.num_procs, order, use_deadline, False))
    return sum(ret)


if __name__ == "__main__":
    util_range = get_util_range(args.num_procs)
    tesets = []
    on = False
    for util in util_range:
        on = False
        if util == args.range_l:
            on = True
        if on:
            with open("../gfpsdata/te/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                tesets.append(ts)
        if util == args.range_r:
            break

    test_dataset = Datasets(tesets)
    test_dataset.setlen(args.num_test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False
    )

    """Heuristic Test"""
    print("OPA")
    test_heu(test_dataset, "OPA")
    print()

    """Distillation Model Test"""
    print("LINEAR")
    dist_file_name = "LIN-p%d-t%d-d%d-l[%s, %s]" % (
        args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    Distillation = LinearSolver(args.num_procs, args.num_tasks,
                               args.use_deadline, False)
    with open("../gfpsmodels/distillation/" + dist_file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f)
    Distillation.load_state_dict(tmp.state_dict())
    Distillation.cpu()
    Distillation.eval()
    test_distillation(Distillation, test_loader)
    print()

    """Reinforcement Learning Model Test"""
    print("Global REINFORCE")
    global_file_name = "globalRL-p%d-t%d-d%d-l" % (
        args.num_procs, args.num_tasks, int(use_deadline)
    )
    GlobalRlModel = Solver(args.num_procs, args.embedding_size, args.hidden_size,
                           args.num_tasks, use_deadline=False, use_cuda=False).cpu()
    with open("../gfpsmodels/globalrl/" + global_file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f)
    GlobalRlModel.load_state_dict(tmp.state_dict())
    GlobalRlModel.eval()
    test_global_reinforce(GlobalRlModel, test_loader)
    print()

    print("Local REINFORCE")
    rl_file_name = "localRL-p%d-t%d-d%d-l[%s, %s]" % (
        args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    RLModel = Solver(args.num_procs, args.embedding_size, args.hidden_size,
                     args.num_tasks, use_deadline=False, use_cuda=False).cpu()
    with open("../gfpsmodels/localrl/" + rl_file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f)
    RLModel.load_state_dict(tmp.state_dict())
    RLModel.eval()
    test_reinforce(RLModel, test_loader)
    print()

    """GumbelSearch Model Test"""
    print("GUMBELSEARCH")
    test_gumbel(Distillation, test_loader, 10)
    print()

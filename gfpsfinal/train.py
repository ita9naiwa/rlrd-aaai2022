import time
import os
import argparse
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.stats
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import cy_heuristics as heu
import sched_heuristic as py_heu
from sched_solver import Solver
from util import Datasets, get_util_range, load_datasets

parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=32)
parser.add_argument("--num_procs", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--num_train_dataset", type=int, default=200000)
parser.add_argument("--num_test_dataset", type=int, default=5000)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--lr", type=float, default=1.0 * 1e-4)
parser.add_argument("--lr_decay_step", type=int, default=100)
parser.add_argument("--use_deadline", type=int, default=0)


confidence = 0.05

args = parser.parse_args()
use_deadline = args.use_deadline
test_module = heu.test_RTA_LC
use_cuda = True

if __name__ == "__main__":
    directory = os.path.dirname("../gfpsmodels/globalrl/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    if use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False

    util_range = get_util_range(args.num_procs)
    trsets, tesets = load_datasets(args.num_procs, args.num_tasks)

    train_dataset = Datasets(trsets)
    test_dataset = Datasets(tesets)

    train_dataset.setlen(args.num_train_dataset)
    test_dataset.setlen(args.num_test_dataset)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculating heuristics
    temp_fname = "%s-%s-%s" % (args.num_procs, args.num_tasks, args.use_deadline)
    require_pt = False
    try:
        model = torch.load("models/" + temp_fname).cuda()
        model.train()
    except:
        require_pt = True
        print("No previous model!")
        model = Solver(args.num_procs,
                   args.embedding_size,
                   args.hidden_size,
                   args.num_tasks,
                   use_deadline=False,
                   use_cuda=use_cuda)
    bl_model = Solver(args.num_procs,
                   args.embedding_size,
                   args.hidden_size,
                   args.num_tasks,
                   use_deadline=False,
                   use_cuda=use_cuda)
    bl_model.load_state_dict(model.state_dict())
    if use_cuda:
        model = model.cuda()
        bl_model = bl_model.cuda()

    bl_model = bl_model.eval()

    def wrap(x):
        _sample, num_proc, use_deadline = x
        return heu.OPA(_sample, num_proc, None, use_deadline)       # 여기서 OPA에 테스트 안넘겼는데 까고 들어가보면 DA어쩌구 테스트를 사용함.

    with ProcessPoolExecutor(max_workers=10) as executor:
        inputs = []
        res_opa = np.zeros(len(test_dataset), dtype=int).tolist()
        for i, sample in test_dataset:
            inputs.append((sample, args.num_procs, use_deadline))
        for i, ret in tqdm(enumerate(executor.map(wrap, inputs))):
            res_opa[i] = ret
        opares = np.sum(res_opa)

    start = time.time()
    print("[before training][OPA generates %d]" % opares)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.5 * args.lr)
    loss_ = 0
    avg_hit = []
    updates = 0
    prev_ = -1
    require_pt = True
    if require_pt:
        for batch_idx, (_, sample_batch) in enumerate(train_data_loader):
            guide = []
            as_np = sample_batch.numpy()
            for r in as_np:
                guide.append(np.argsort(py_heu.get_DM_scores(r, args.num_procs, use_deadline)))
            guide = torch.from_numpy(np.array(guide, dtype=np.int64))
            if use_cuda:
                guide = guide.cuda()
                sample_batch = sample_batch.cuda()
            num_samples = sample_batch.shape[0]
            optimizer.zero_grad()
            rewards, log_probs, action = model(sample_batch, guide=guide)
            advantage = rewards
            if use_cuda:
               advantage = advantage.cuda()
            loss = -torch.sum((advantage * log_probs), dim=-1).mean()
            loss.backward()
            loss_ += loss.cpu().detach().numpy()
            avg_hit.append((rewards.cpu().detach().mean()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            updates += 1
            if (updates % 100 == 0):
                model.eval()
                ret = []
                for i, _batch in eval_loader:
                    if use_cuda:
                        _batch = _batch.cuda()
                    R, log_prob, actions = model(_batch, argmax=True)
                    for j, chosen in enumerate(actions.cpu().numpy()):
                        order = np.zeros_like(chosen)
                        for i in range(args.num_tasks):
                            order[chosen[i]] = args.num_tasks - i - 1      # 먼저 뽑히면 우선순위 높다.
                        if use_cuda:
                            ret.append(test_module(_batch[j].cpu().numpy(), args.num_procs, order, use_deadline, False))
                        else:
                            ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
                rl_model_sum = np.sum(ret)
                print("[consumed %d samples][PRETRAINING][RL model generates %d][OPA generates %d]" % (
                updates * args.batch_size, rl_model_sum, opares), "log_probability\t",
                      log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit))
                if rl_model_sum <= prev_ or updates >= 50:
                    print("RL pretraining end")
                    break
                prev_ = rl_model_sum
                model.train()
    model.train()
    bl_model.load_state_dict(model.state_dict())
    bl_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=0.9, last_epoch=-1)
    last_rl_model_sum = -1
    updates = 0
    noupdateinarow = 0
    _max = -1
    for epoch in range(args.num_epochs):
        loss_ = 0
        avg_hit = []
        for batch_idx, (_, sample_batch) in enumerate(train_data_loader):
            if use_cuda:
                sample_batch = sample_batch.cuda()
            num_samples = sample_batch.shape[0]
            optimizer.zero_grad()
            rewards, log_probs, action = model(sample_batch)
            baseline, _bl_log_probs, _bl_action = bl_model(sample_batch, argmax=True)
            advantage = rewards - baseline
            if use_cuda:
                advantage = advantage.cuda()
            loss = -torch.sum((advantage * log_probs), dim=-1).mean()
            loss.backward()
            loss_ += loss.cpu().detach().numpy()
            avg_hit.append((rewards.cpu().detach().mean()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            updates += 1
            if use_cuda:
                diff = advantage.sum(dim=-1).detach().cpu().numpy()
            else:
                diff = advantage.sum(dim=-1).detach().numpy()
            D = diff.mean()
            S_D = 1e-10 + np.sqrt(((diff - D) ** 2).sum() / (1e-10 + num_samples - 1))
            tval = D / (S_D / (1e-10 + math.sqrt(1e-10 + num_samples)))
            p = scipy.stats.t.cdf(tval, num_samples)
            if (p >= 1. - 0.5 * confidence) or (p <= 0.5 * confidence):
                bl_model.load_state_dict(model.state_dict())
            if updates % 100 == 0:
                model.eval()
                ret = []
                for i, _batch in eval_loader:
                    if use_cuda:
                        _batch = _batch.cuda()
                    R, log_prob, actions = model(_batch, argmax=True)
                    for j, chosen in enumerate(actions.cpu().numpy()):
                        order = np.zeros_like(chosen)
                        for i in range(args.num_tasks):
                            order[chosen[i]] = args.num_tasks - i - 1       #중요할수록 숫자가 높다.
                        if use_cuda:
                            ret.append(test_module(_batch[j].cpu().numpy(), args.num_procs, order, use_deadline, False))
                        else:
                            ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
                fname = "globalRL-p%d-t%d-d%d-l" % (args.num_procs, args.num_tasks, int(use_deadline))
                rl_model_sum = np.sum(ret)

                end = time.time()
                elapsed = (end - start)
                minute = int(elapsed // 60)
                second = int(elapsed - 60 * minute)

                print("경과시간 : {}m {}s".format(minute, second))
                print("[consumed %d samples][at epoch %d][RL model generates %d][OPA generates %d]"
                      % (updates * args.batch_size, epoch, rl_model_sum, opares),
                      "log_probability\t", log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit))
                torch.save(model, "../gfpsmodels/globalrl/" + fname + ".torchmodel")
                stop = False
                # with open("globallog/" + fname, 'a') as f:
                #     print("[consumed %d samples][at epoch %d][RL model generates %d][OPA generates %d]" % (updates * args.batch_size, epoch, rl_model_sum, opares),
                #           "log_probability\t", log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit), file=f)
                if rl_model_sum == args.num_test_dataset:
                    print("경과시간 : {}m {}s".format(minute, second))
                    print("total hit at epoch", epoch)
                    print("SAVE SUCCESS")
                    stop = True
                if rl_model_sum > _max:
                    noupdateinarow = 0
                    _max = rl_model_sum
                    print("경과시간 : {}m {}s".format(minute, second))
                    print("SAVE SUCCESS")
                else:
                    noupdateinarow += 1
                if noupdateinarow >= 20:
                    print("경과시간 : {}m {}s".format(minute, second))
                    print("not update m0 times", epoch)
                    print("SAVE SUCCESS")
                    stop = True
                if stop:
                    raise NotImplementedError

            model.train()

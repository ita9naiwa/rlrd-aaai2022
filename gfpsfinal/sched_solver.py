import pyximport
pyximport.install()
import math
from copy import copy
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from rl_with_attention import AttentionTSP
import sched_heuristic as heu
import cy_heuristics as cy_heu
import sched
from sched_heuristic import scores_to_priority
from concurrent.futures import ProcessPoolExecutor

INPUT_SIZE = 10
N_WORKERS = 8

import scipy.signal as signal
def get_return(rewards, discount=1.0):
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    y = signal.lfilter(b, a, x=r)
    return y[::-1]
def wrap(x):
    (i, j), y = x
    rets = []

    for x in y:
        _sample, num_proc, order, use_deadline = x
        score, rews = cy_heu.test_RTA_LC(_sample, num_proc, order, use_deadline=use_deadline, ret_score=2)
        if score < 0:
            score = 0
        rets.append(get_return(rews) + score)
    return (i, j), rets

def wrap_np(x):
    (i, j), y = x
    rets = []

    for x in y:
        _sample, num_proc, order, use_deadline = x
        score , ret = cy_heu.test_Lee(_sample, num_proc, order, use_deadline=use_deadline, ret_score=2)
        rets.append(get_return(ret) + score)
    return (i, j), rets
def random_shuffle(model, np_single_input, sample_size=256):
    inputs = torch.from_numpy(np_single_input).unsqueeze(0).repeat(sample_size, 1, 1)
    seq_len = inputs.shape[1]
    prob, chosen = model.forward(inputs, get_reward=False)
    tmp = np.arange(sample_size)
    order = np.zeros_like(chosen)
    for i in range(seq_len):
        order[tmp, chosen[:, i]] = seq_len - i - 1
    return order

class BeamNode():
    def __init__(self, query, mask, parents, val=0.0, step=0, comp='default'):
        self.query = query
        self.mask = mask
        self.parents = parents
        self.val = val
        self.step = step
        self.comp = comp
    def decompose(self,):
        return self.query, self.mask, self.parents, self.val, self.step
    def __lt__(self, other):
        if self.val < other.val:
            return True
        else:
            return False

    def __repr__(self):
        return         "Node with val %0.2f, size %d" % (self.val , self.step)


def beam_search(model, np_single_input, beam_size=3):
    bs = beam_size
    inputs = model.input_transform(torch.from_numpy(np_single_input).unsqueeze(0))
    batch_size, seq_len, _ = inputs.shape

    actor = model.actor
    embedded, h, h_mean, h_bar, h_rest, query = model.actor._prepare(inputs)
    nodes = [BeamNode(query=query,
                      mask=torch.zeros(1, seq_len, dtype=torch.bool),
                      parents=[],
                      val=0.0,
                      step=0)]
    for i in range(seq_len):
        r = []
        for candidate in nodes:
            r.extend(beam_next_nodes(actor, candidate, bs, h, h_mean, h_bar, h_rest))
        if len(r) > bs:
            nodes = [r[x] for x in np.argpartition(r, bs)[:bs]]
        else:
            nodes = r
    return [node.parents for node in nodes]


def beam_next_nodes(actor, node, beam_size, h, h_mean, h_bar, h_rest):
    bs = beam_size
    query, mask, parents, val, step = node.decompose()
    _, n_query = actor.glimpse(query, h, mask)
    prob, _  = actor.pointer(n_query, h, mask)
    prob = prob.squeeze(0)
    xxx = prob.topk(min(bs, len(prob)))
    indices = xxx.indices
    log_probs = -torch.log(xxx.values)
    newmask = mask.repeat(bs, 1)
    newparents = [copy(parents) for x in range(bs)]
    newvals = val + log_probs
    newqueries = []
    for i in range(min(bs, len(prob))):
        idx = indices[i]
        newmask[i, idx] = True
        newparents[i].append(idx.detach().numpy())
        chosen_h = h[0, idx]
        h_rest = actor.v_weight_embed(chosen_h)
        newqueries.append(actor.h_query_embed(h_bar + h_rest))

    newNodes = []
    for i in range(min(bs, len(prob))):
        newNodes.append(
            BeamNode(
                query=newqueries[i],
                mask=newmask[i].unsqueeze(0),
                parents=newparents[i],
                val=newvals[i],
                step=step + 1,
                )
        )
    return newNodes


executor = executor = ProcessPoolExecutor(max_workers=N_WORKERS)

class Solver(nn.Module):
    def __init__(self, num_proc, embedding_size, hidden_size, seq_len, tanh_exploration=5, use_deadline=False, use_cuda=True, ret_dist=False):
        super(Solver, self).__init__()
        self.num_proc = num_proc
        self.use_deadline = use_deadline
        self.use_cuda = use_cuda
        print("use_deadline", self.use_deadline)
        print("is_cuda", self.use_cuda)
        self.actor = AttentionTSP(INPUT_SIZE, embedding_size, hidden_size, seq_len, n_head=4, C=tanh_exploration, use_cuda=use_cuda, ret_dist=ret_dist)
        self.ret_dist = ret_dist

    def reward(self, sample, chosen):
        """
        Args:
            sample_solution seq_len of [batch_size]
            torch.LongTensor [batch_size x seq_len x INPUT_SIZE]
        """

        batch_size, seq_len, _ = sample.size()

        rewardarr = torch.FloatTensor(batch_size, seq_len)

        tmp = np.arange(batch_size)
        order = np.zeros_like(chosen)
        for i in range(seq_len):
            order[tmp, chosen[:, i]] = seq_len - i - 1
        _sample = sample.cpu()
        _samples = [(sample, self.num_proc, _order, self.use_deadline) for (sample, _order) in zip(_sample.numpy(), order)]
        tmps = []
        step = (batch_size // N_WORKERS) + 1
        chunks = []
        for i in range(0, batch_size, step):
            chunks.append(((i, min(i+step, batch_size)), _samples[i:min(i+step, batch_size)]))
        for ((i, j), ret) in executor.map(wrap, chunks):
            for q, _ in enumerate(range(i, j)):
                rewardarr[_, :] = torch.from_numpy(ret[q])
        if self.use_cuda:
            ewardarr = rewardarr.cuda()        # [batch_size x seq_len]
        return rewardarr

    def reward_np(self, sample, chosen):
            """
            Args:
                sample_solution seq_len of [batch_size]
                torch.LongTensor [batch_size x seq_len x INPUT_SIZE]
            """

            batch_size, seq_len, _ = sample.size()
            rewardarr = torch.FloatTensor(batch_size, seq_len)
            tmp = np.arange(batch_size)
            order = np.zeros_like(chosen)
            for i in range(seq_len):
                order[tmp, chosen[:, i]] = seq_len - i - 1
            _sample = sample.cpu()
            _samples = [(sample, self.num_proc, _order, self.use_deadline) for (sample, _order) in zip(_sample.numpy(), order)]
            tmps = []
            step = (batch_size // N_WORKERS) + 1
            chunks = []
            for i in range(0, batch_size, step):
                chunks.append(((i, min(i+step, batch_size)), _samples[i:min(i+step, batch_size)]))
            for ((i, j), ret) in executor.map(wrap_np, chunks):
                for q, _ in enumerate(range(i, j)):
                    rewardarr[_, :] = torch.from_numpy(ret[q])
            if self.use_cuda:
                rewardarr = rewardarr.cuda()
            #print(rewardarr)
            return rewardarr

    def input_transform(self, inputs):
        """
        Args:¡
            inputs: [batch_size, seq_len, 3]
        """

        if self.use_deadline is False:
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
        if self.use_cuda:
            ret = ret.cuda()

        return ret

    def forward(self, inputs, argmax=False, get_reward=True, guide=None, multisampling = False):
        """
        Args:
            inputs: [batch_size, seq_len, 3] (T, C, D)
        """
        batch_size, seq_len, _ = inputs.shape

        _inputs = self.input_transform(inputs)
        if self.ret_dist:
            return self.actor(_inputs, argmax, guide=guide)
        probs, actions = self.actor(_inputs, argmax, guide=guide, multisampling = multisampling)
        # if self.ret_score:
        #     return probs

        if get_reward:
            R = self.reward(inputs, actions.cpu().numpy())  # [batch_size x seq_len]
            return R, probs, actions
        else:
            return probs, actions

    def forward_np(self, inputs, argmax=False, get_reward=True, guide=None):
        """
        Args:
            inputs: [batch_size, seq_len, 3] (T, C, D)
        """
        batch_size, seq_len, _ = inputs.shape

        _inputs = self.input_transform(inputs)

        probs, actions = self.actor(_inputs, argmax, guide=guide)
        #print(probs)
        if get_reward:
            R = self.reward_np(inputs, actions.cpu().numpy())

            return R, probs, actions
        else:
            return probs, actions

    def multisampling(self, inputs):
        actions = self.actor.multisampling(inputs)
def test():
    seq_len = 20
    solver = Solver(64, 64, seq_len)
    ds = sched.SchedSingleDataset(seq_len, 1000)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    for idx, dps in loader:
        solver.forward(dps)

if __name__ == "__main__":
    test()

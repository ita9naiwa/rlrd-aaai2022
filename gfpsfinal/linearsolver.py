import scipy.signal as signal
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import cy_heuristics as cy_heu
import sched

INPUT_SIZE = 10
N_WORKERS = 8

executor = ProcessPoolExecutor(max_workers = N_WORKERS)


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


class LinearSolver (nn.Module):
    def __init__(self, num_proc, seq_len, use_deadline=False,
                 use_cuda=True, hidden_size=256):
        super(LinearSolver, self).__init__()
        self.num_proc = num_proc
        self.seq_len = seq_len
        self.use_deadline = use_deadline
        self.use_cuda = use_cuda
        self.layer = nn.Linear(10, 1)
        nn.init.uniform_(self.layer.weight, 0, 1)

        if self.use_deadline:
            print("EXPLICIT MODE")
        else:
            print("IMPLICIT MODE")

    def input_transform(self, inputs):
        """
        Args:¡
            inputs: [batch_size, seq_len, 3]
        """

        if self.use_deadline is False:      # implicit
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

    def forward(self, inputs, normalize=False):
        """
        Args:
            inputs : [batch_size, seq_len, 3] : Before the transformation.
        Returns:
            score : [batch_size, seq_len]
        """

        batch_size, seq_len, _ = inputs.shape
        inputs_tr = self.input_transform(inputs)
        score = self.layer(inputs_tr).squeeze()
        if normalize:
            score = torch.log(torch.softmax(score, dim=-1))
            # print(score)
            return score
        return score

    def ranknet_loss(self, inputs, labels):
        """
        :param inputs: [batch_size, seq_len, 3]
        :param labels: [batch_size, seq_len]
        """
        scores = self.forward(inputs)
        pred_mtx, label_mtx = self.convert_to_matrix(scores, labels)
        pred_mtx = F.sigmoid(pred_mtx)
        loss = torch.sum(-label_mtx * torch.log(pred_mtx) - (1 - label_mtx) * torch.log(1 - pred_mtx))
        return loss

    def convert_to_matrix(self, scores, labels):
        """
        :param scores: [batch_size, seq_len]
        :param labels: [batch_size, seq_len]
        """
        batch_size, seq_len = scores.size()
        scores = nn.functional.normalize(scores, dim=1)
        score_ret = torch.zeros((batch_size, seq_len, seq_len))
        label_ret = torch.zeros((batch_size, seq_len, seq_len))
        # scores /= 10000
        for i in range(seq_len):
            for j in range(seq_len):
                score_ret[:, i, j] = scores[:, i] - scores[:, j]
                label_ret[:, i, j] = labels[:, i] > labels[:, j]
        return score_ret, label_ret

    def listnet_loss(self, inputs, labels, phi="softmax"):
        """
        :param inputs: [batch_size, seq_len, 3]
        :param labels: [batch_size, seq_len]
        """
        scores = self.forward(inputs)       # [batch_size x seq_len]
        labels = labels.float()
        if phi == "softmax":
            pred = torch.softmax(scores, -1)      # [batch_size x seq_len]
            target = torch.softmax(labels, -1)
        elif phi == "log":
            pred = torch.log(scores, -1) / torch.sum(torch.log(scores, -1))
            target = torch.log(labels, -1) / torch.sum(torch.log(labels, -1))
        elif phi.startswith("id"):
            pred = scores / torch.sum(scores, -1)
            target = labels / torch.sum(labels)
        else:
            raise LookupError("WHAT INCREASING FUNCTION YOU NEED")
        # pred, target : [batch_size x seq_len]
        loss = -torch.sum(target * torch.log(pred))
        return loss




class LinearActor(nn.Module):
    def __init__(self, input_dim, use_cuda=True):
        super(LinearActor, self).__init__()
        self.use_cuda = use_cuda
        self.layer = nn.Linear(input_dim, 1)

    def forward(self, inputs, argmax=False, guide=None):
        """
        Args:
             inputs : [batch_size x seq_len x input_dim]
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        mask = torch.zeros(batch_size, seq_len, dtype = torch.bool)
        for index in range(seq_len):

            prob = self.pointer(inputs, mask)
            cat = Categorical(prob)
            if argmax:
                _, chosen = torch.max(prob, -1)
            elif guide is not None:
                chosen = guide[:, index]
            else:
                chosen = cat.sample()
            logprobs = cat.log_prob(chosen)
            prev_chosen_logprobs.append(logprobs)
            prev_chosen_indices.append(chosen)

            mask[[i for i in range(batch_size)], chosen] = True

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)

    def pointer(self, inputs, mask):
        """
        Args:
             mask : [batch_size x seq_len]
        """
        score = self.layer(inputs).squeeze()
        score[mask] = -1e8
        score = torch.softmax(score, dim=-1)
        return score


class LinearRLSolver(nn.Module):
    def __init__(self, num_proc,
                 use_deadline = False,
                 use_cuda = True):
        super(LinearRLSolver, self).__init__()
        self.num_proc = num_proc
        self.use_deadline = use_deadline
        self.use_cuda = use_cuda
        print("use_deadline", self.use_deadline)
        print("is_cuda", self.use_cuda)
        self.actor = LinearActor(input_dim=10, use_cuda=use_cuda)

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
        _samples = [(sample, self.num_proc, _order, self.use_deadline) for (sample, _order) in
                    zip(_sample.numpy(), order)]
        tmps = []
        step = (batch_size // N_WORKERS) + 1
        chunks = []
        for i in range(0, batch_size, step):
            chunks.append(((i, min(i + step, batch_size)), _samples[i:min(i + step, batch_size)]))
        for ((i, j), ret) in executor.map(wrap, chunks):
            for q, _ in enumerate(range(i, j)):
                rewardarr[_, :] = torch.from_numpy(ret[q])
        if self.use_cuda:
            rewardarr = rewardarr.cuda()
        return rewardarr

    def input_transform(self, inputs):
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

    def forward(self, inputs, argmax=False, get_reward=True, guide=None):
        """
                Args:
                    inputs: [batch_size, seq_len, 3] (T, C, D)
                """
        batch_size, seq_len, _ = inputs.shape
        _inputs = self.input_transform(inputs)
        probs, actions = self.actor(_inputs, argmax, guide=guide)
        if get_reward:
            R = self.reward(inputs, actions.cpu().numpy())
            return R, probs, actions
        else:
            return probs, actions


def sample_gumbel(score, sampling_number=5, eps=1e-10):
    """
    Args:
         score : [batch x num_tasks]
    """

    tmax, _ = torch.max(score, dim=-1)
    tmin, _ = torch.min(score, dim=-1)

    score = score / (tmax - tmin).unsqueeze(1)
    score = score * score.size(1)
    batch_size = score.size(0)
    num_tasks = score.size(1)
    U = torch.rand([batch_size, sampling_number, num_tasks])
    return score.unsqueeze(1) - torch.log(-torch.log(U + eps) + eps)


def get_rank(score):
    """
    Args:
         score:  [batch_size x num_gumbel_sample x num_tasks]
    """
    num_tasks = score.size(2)
    arg = torch.argsort(-score, dim=-1).numpy()       # [batch_size x num_gumbel_sample x num_tasks]
    ret = np.zeros_like(arg)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            for k in range(ret.shape[2]):
                ret[i][j][arg[i, j, k]] = num_tasks - k - 1
    return ret

import copy
import time
import numpy as np
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_rank
from MDKP_env import Env
from network import Att_Policy
from hyperparams import *
from concurrent.futures import ProcessPoolExecutor as PExec

_exec = PExec(1)
EPS = 1e-8
ALPHA = 0.3

def flatten(_list):
    return [i for i in _list]
    # print([i for i in _list])
    # return [i for _sublist in _list for i in _sublist]

def timer(func):
    @wraps(func)
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        print("Function {name}, Time : {time:.5f}"
              .format(name=func.__name__, time=end-start))
        return result
    return wrapper


def get_discounted_returns(rewards):
    ret = []
    R = 0
    for r in rewards[::-1]:
        R = r + args.discount_factor * R
        ret.insert(0, R)
    return ret


def get_greedy_returns(items, knapsacks):
    eplen, x = simplest_greedy(items, knapsacks)
    y = np.zeros(eplen)
    y[-1] = x
    return get_discounted_returns(y)


def get_env(item_dim, num_items, num_knapsacks):
    return Env(item_dim, num_items, num_knapsacks)


def get_model(dimension, embed_dim, use_cuda):
    # return Linear_Policy(dimension, use_cuda)
    return Att_Policy(dimension, embed_dim, use_cuda)


def multi_rollout(env, policy, parameter_queue, step_size):
    average_value = 0
    average_return = []
    returns = []
    logprobs = []
    baselines = np.zeros((step_size, 1000))
    for batch in range(step_size):
        observation, _ = env.reset()
        policy.reset()
        step = 0
        while 1:
            step += 1
            items, knapsack, allocable_items, allocable_knapsacks = observation
            if args.use_cuda:
                items = torch.from_numpy(items).float().to("cuda:0")
                knapsack = torch.from_numpy(knapsack).float().to("cuda:0")
            else:
                items = torch.from_numpy(items).float()
                knapsack = torch.from_numpy(knapsack).float()
            item, knapsack, log_prob = policy(items, knapsack, allocable_items, allocable_knapsacks, True)
            next_observation, reward, done = env.step(item, knapsack)
            if done:
                break
            observation = next_observation
        discounted_returns = get_discounted_returns(env.rewards)
        returns.append(discounted_returns)
        average_return.append(np.sum(discounted_returns))
        baselines[batch, : len(discounted_returns)] = discounted_returns
        logprobs.append(torch.stack(policy.log_probs, dim=-1))
        average_value += env.total_value()
    baseline = np.mean(baselines, axis=0).reshape(1, -1)
    loss = 0.0
    average_value /= step_size
    for i in range(step_size):
        ret = np.array(returns[i]).reshape(1, -1)
        adv = torch.from_numpy(ret - baseline[0, :ret.shape[1]]).float()
        lp = logprobs[i]
        if args.use_cuda:
            lp = lp.to("cpu")
        loss += torch.sum(-(adv * lp))

    loss /= step_size
    loss.backward()
    gradient = [x.grad for x in policy.parameters()]
    ret = (gradient, np.array(average_return), loss.detach().numpy(), average_value)
    parameter_queue.put(ret)
    return ret


def rollout(env, policy, guide=False):
    average_value = 0
    average_return = 0
    returns = []
    logprobs = []
    baselines = np.zeros((args.batch_size, 1000))
    timestep = 0
    for batch in range(args.batch_size):
        observation, item_amounts = env.reset()
        policy.reset()
        while 1:
            timestep += 1
            items, knapsacks, allocable_items, allocable_knapsacks = observation
            if args.use_cuda:
                _items = torch.from_numpy(items).float().to("cuda:0")
                _knapsacks = torch.from_numpy(knapsacks).float().to("cuda:0")
            else:
                _items = torch.from_numpy(items).float()
                _knapsacks = torch.from_numpy(knapsacks).float()
            item, log_prob = policy(_items, _knapsacks, allocable_items,
                                              allocable_knapsacks, guide=guide)

            next_observation, reward, done = env.step(item, 0)
            if done:
                break
            observation = next_observation
        discounted_returns = get_discounted_returns(env.rewards)

        returns.append(discounted_returns)
        average_return += np.sum(discounted_returns)

        greedy_returns = get_greedy_returns(items, knapsacks)
        baselines[batch, :len(greedy_returns)] = greedy_returns
        # baselines[batch, : len(discounted_returns)] = discounted_returns      # Mean greedy
        logprobs.append(torch.stack(policy.log_probs, dim=-1))
        average_value += env.total_value()
    baseline = np.mean(baselines, axis=0).reshape(1, -1)        # [1 x 1000]
    loss = 0.0
    for i in range(args.batch_size):
        ret = np.array(returns[i]).reshape(1, -1)
        # adv = torch.from_numpy(ret)
        adv = torch.from_numpy(ret - baseline[0, :ret.shape[1]]).float().squeeze()
        lp = logprobs[i]
        if args.use_cuda:
            lp = lp.to("cpu")
        loss += torch.sum(-(adv * lp))

    loss /= (args.batch_size * timestep)
    loss.backward()
    return loss, average_value / args.batch_size, average_return / args.batch_size


def train_distilation(env, teacher, student, criterion, zeroone_loss=False):
    num_items = env.num_items
    cumulated_loss = 0
    for batch in range(args.batch_size):
        observation, _ = env.reset()
        teacher.reset()
        student.reset()
        items, knapsacks, _, _ = observation
        if args.use_cuda:
            items = torch.from_numpy(items).float().to("cuda:0")
            knapsacks = torch.from_numpy(knapsacks).float().to("cuda:0")
        else:
            items = torch.from_numpy(items).float()
            knapsacks = torch.from_numpy(knapsacks).float()
        linear_score = student(items, knapsacks)
        selected_pair = []
        while 1:
            items, knapsacks, allocable_items, allocable_knapsacks = observation
            if args.use_cuda:
                items = torch.from_numpy(items).float().to("cuda:0")
                knapsacks = torch.from_numpy(knapsacks).float().to("cuda:0")
            else:
                items = torch.from_numpy(items).float()
                knapsacks = torch.from_numpy(knapsacks).float()
            item, log_prob = teacher(
                items, knapsacks, allocable_items,
                allocable_knapsacks, True
            )
            selected_pair.append((item, 0))
            next_observation, reward, done = env.step(item, 0)
            if done:
                break
            observation = next_observation
        rl_item_order = torch.zeros((1, num_items))
        for j in range(len(selected_pair)):
            rl_item_order[0][selected_pair[j][0]] = j + 1
        linear_score = linear_score.view(1, -1)
        if zeroone_loss:
            zero_one_loss = calculate_zero_one(linear_score, rl_item_order, items, knapsacks)
        if args.use_cuda:
            linear_soft_rank = soft_rank(
                linear_score.cpu(), regularization_strength=0.0001, direction="DESCENDING"
            )
        else:
            linear_soft_rank = soft_rank(
                linear_score, regularization_strength=0.0001, direction="DESCENDING"
            )
        ref = []

        for i in range(len(selected_pair)):
            ref.append(selected_pair[i][0])

        mask = []
        for i in range(num_items):
            if i not in ref:
                mask.append(i)
        linear_soft_rank[0][mask] = 0
        rank_loss = criterion(rl_item_order, linear_soft_rank)
        if zeroone_loss:
            rank_loss = rank_loss.to("cuda:0")
            zero_one_loss = zero_one_loss.to("cuda:0")
            loss = (ALPHA * rank_loss) + ((1 - ALPHA) * zero_one_loss)
        else:
            loss = rank_loss
        cumulated_loss += loss
    return cumulated_loss / args.batch_size


def ranknet_loss(env, teacher, student, criterion):
    num_items = env.num_items
    for batch in range(args.batch_size):
        observation, _ = env.reset()
        teacher.reset()
        student.reset()
        items, knapsacks, _, _ = observation
        if args.use_cuda:
            items = torch.from_numpy(items).float().to("cuda:0")
            knapsacks = torch.from_numpy(knapsacks).float().to("cuda:0")
        else:
            items = torch.from_numpy(items).float()
            knapsacks = torch.from_numpy(knapsacks).float()
        linear_score = student(items, knapsacks)
        selected_pair = []
        while 1:
            items, knapsacks, allocable_items, allocable_knapsacks = observation
            if args.use_cuda:
                items = torch.from_numpy(items).float().to("cuda:0")
                knapsacks = torch.from_numpy(knapsacks).float().to("cuda:0")
            else:
                items = torch.from_numpy(items).float()
                knapsacks = torch.from_numpy(knapsacks).float()
            item, log_prob = teacher(
                items, knapsacks, allocable_items,
                allocable_knapsacks, True
            )
            selected_pair.append(item)
            next_observation, reward, done = env.step(item, 0)
            if done:
                break
            observation = next_observation
        rl_item_order = torch.ones((1, num_items)) * num_items
        for j in range(len(selected_pair)):
            rl_item_order[0][selected_pair[j]] = j + 1      # 랭크
        linear_score = linear_score.view(1, -1)
        pred_mtx, label_mtx = convert_to_matrix(linear_score, rl_item_order)
        pred_mtx = F.sigmoid(pred_mtx)
        loss = torch.sum(-label_mtx * torch.log(pred_mtx) - (1 - label_mtx) * torch.log(1 - pred_mtx))
        return loss


def convert_to_matrix(scores, label):
    """
    :param scores: [1 x num_items]
    :param label: [1 x num_items]
    """
    num_items = scores.size(1)
    scores = nn.functional.normalize(scores, dim=1)
    score_ret = torch.zeros((num_items, num_items))
    label_ret = torch.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            score_ret[i, j] = scores[0, i] - scores[0, j]
            label_ret[i, j] = label[0, i] < label[0, j]
    return score_ret, label_ret


def listnet_loss(env, teacher, student, criterion):
    num_items = env.num_items
    observation, _ = env.reset()
    teacher.reset()
    student.reset()
    items, knapsacks, _, _ = observation
    if args.use_cuda:
        items = torch.from_numpy(items).float().to("cuda:0")
        knapsacks = torch.from_numpy(knapsacks).float().to("cuda:0")
    else:
        items = torch.from_numpy(items).float()
        knapsacks = torch.from_numpy(knapsacks).float()
    linear_score = student(items, knapsacks)
    selected_pair = []
    while 1:
        items, knapsacks, allocable_items, allocable_knapsacks = observation
        if args.use_cuda:
            items = torch.from_numpy(items).float().to("cuda:0")
            knapsacks = torch.from_numpy(knapsacks).float().to("cuda:0")
        else:
            items = torch.from_numpy(items).float()
            knapsacks = torch.from_numpy(knapsacks).float()
        item, log_prob = teacher(
            items, knapsacks, allocable_items,
            allocable_knapsacks, True
        )
        selected_pair.append(item)
        next_observation, reward, done = env.step(item, 0)
        if done:
            break
        observation = next_observation
    rl_item_order = torch.zeros((1, num_items))
    for j in range(len(selected_pair)):
        rl_item_order[0][selected_pair[j]] = j + 1  # 랭크
    mask = []
    for i in range(num_items):
        if rl_item_order[0][i] == 0:
            mask.append(i)
    for i in range(len(mask)):
        # linear_score[0][mask[i]] = -1e8
        rl_item_order[0][mask[i]] = -1e8
    prediction = torch.softmax(linear_score, -1)
    target = torch.softmax(rl_item_order, -1)
    loss = -torch.sum(target * torch.log(prediction))
    return loss


def sample_gumbel(score, sampling_number=500, eps=1e-10):
    """
    Score : [1 x num_items]
    """
    score = torch.log(torch.softmax(score, -1))

    tmax, _ = torch.max(score, dim=-1)
    tmin, _ = torch.min(score, dim=-1)
    #
    score = score / (tmax - tmin).unsqueeze(1)
    score -= score.mean(-1)
    score = score * score.size(1)
    # score = 3.0 * torch.tanh(score)
    num_items = score.size(1)
    U = torch.rand([sampling_number] + [num_items])
    # ma, _ = torch.max(score, dim=-1)
    # mi, _ = torch.min(score, dim=-1)
    Z =  score - torch.log(-torch.log(U + eps) + eps)     # [sampling_number x num_items]
    return Z


def get_rank(score):
    # ret = torch.zeros_like(score, dtype = torch.int32)
    # score = torch.tensor([[0.9, 0.7, 0.1, 0.5, 1, 0.0]])
    ret = torch.argsort(-score, dim=-1)                       # [4 0 1 3 2 5]
    ret = ret.numpy()
    return ret


def Gumbel_Search2(items, knapsacks, gumbel_rank, getlen=False):
    """
    Args:
        items : [num_items x 1 + k], +1 for value of items
        gumbel_rank : output of get_rank : [sampling_number x num_items]
        knapsacks : [1 x k]
    """

    # rets = []
    # for i in range(gumbel_rank.shape[0]):       # ~ num_sampling
    #     tmp = np.zeros_like(knapsacks)
    #     val = 0
    #     for j in range(gumbel_rank.shape[1]):
    #         if np.sometrue((items[gumbel_rank[i][j], 1:] + tmp) > knapsacks):
    #             break
    #         tmp += items[gumbel_rank[i][j], 1:]
    #         val += items[gumbel_rank[i][j], 0]
    #     rets.append(val)
    # # r = max([_[0] for _ in ret])
    # # idx = [_[0] for _ in ret].index(r)
    # # return ret[idx]
    # return max(rets)

# @timer
def Gumbel_Search2(items, knapsacks, gumbel_rank, getlen=False):
    """
    Args:
        items : [num_items x 1 + item_dim], +1 for value of items
        gumbel_rank : output of get_rank : [sampling_number x num_items]
        knapsacks : [num_knapsacks x item_dim]
    """
    values = items[:, 0]
    bags = items[:, 1:]
    ret = 0
    sh = gumbel_rank.shape[0]
    for i in range(sh):       # ~ num_sampling
        tmp = np.array(knapsacks)
        val = 0
        for j in range(gumbel_rank.shape[1]):
            tmp -= bags[gumbel_rank[i][j]]
            if np.any(tmp < 0):
                break
            val += values[gumbel_rank[i][j]]
        if val > ret:
            ret = val
    return ret


def Gumbel_Search(items, knapsacks, gumbel_rank, getlen=False, num_chunks=16):
    """
    Args:
        items : [num_items x 1 + item_dim], +1 for value of items
        gumbel_rank : output of get_rank : [sampling_number x num_items]
        knapsacks : [num_knapsacks x item_dim]
    """
    values = items[:, 0]
    bags = items[:, 1:]
    ret = 0
    sh = gumbel_rank.shape[0]
    inputs = []
    j = 0
    while j < sh:
        inputs.append((gumbel_rank[j:j+num_chunks], np.array(bags), np.array(knapsacks), np.array(values)))
        j = j + num_chunks
        ret = _exec.map(get_score, inputs)
    ret = flatten(ret)
    return max(ret)


def get_score(input):
    gumbel_rank, bags, knapsacks, values = input
    val = 0
    ret = 0
    n_i, n_j = gumbel_rank.shape
    for i in range(n_i):
        val = 0
        tmp = np.array(knapsacks)
        for j in range(n_j):
            tmp -= bags[gumbel_rank[i][j]]
            if np.any(tmp < 0):
                break
            val += values[gumbel_rank[i][j]]
        if val > ret:
            ret = val
    return ret

# @timer
def simplest_greedy(items, knapsack):
    """
    :param items: [num_items x (1 + item_dim)]
    :param knapsack: [1 x item_dim]
    """
    num_items = items.shape[0]
    value = items[:, 0]
    utils = np.mean(items[:, 1:] / knapsack, axis=-1)
    utils = value / utils
    priority = np.argsort(-utils)
    acc_val = 0
    acc_weight = np.zeros_like(knapsack)
    for i in range(num_items):
        if np.sometrue(acc_weight + items[priority[i], 1:] >= knapsack):
            break
        acc_weight += items[priority[i], 1:]
        acc_val += items[priority[i], 0]
    return i, acc_val


class EarlyStopping:
    # Code from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    # We dont use model save here.
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# @timer
def test_rl(oenv, model, argmax=True):
    env = copy.deepcopy(oenv)
    # observation, item_amounts = env.reset()
    model.reset()
    observation= env.observe()
    x = 0
    while 1:
        x += 1
        items, knapsack, allocable_items, allocable_knapsacks = \
            observation
        items = torch.from_numpy(items).float()
        knapsack = torch.from_numpy(knapsack).float()
        item, log_prob = model(
            items, knapsack, allocable_items,
            allocable_knapsacks, argmax=argmax)
        next_observation, reward, done = env.step(item, 0)
        if done:
            break
        observation = next_observation

    return env.total_value(), x


# @timer
def test_distil(oenv, model):
    env = copy.deepcopy(oenv)
    items, knapsack, allocable_items, allocable_knapsacks = env.observe()       # knapsack : [1 x item_dim], items : [num_items x 1 + item_dim]
    weights = items[:, 1:]
    _items = torch.from_numpy(items).float()
    _knapsack = torch.from_numpy(knapsack).float()
    _linear_selection = model(_items, _knapsack)
    _linear_selection = _linear_selection.detach().numpy()
    _linear_selection = np.argsort(-_linear_selection).squeeze()
    val = 0
    for i in range(1000):
        selection = _linear_selection[i]
        knapsack -= weights[selection]
        if np.any(knapsack < 0):
            break
        val += items[selection][0]
    return val


# @timer
def test_gumbel(oenv, model, sampling_number):
    env = copy.deepcopy(oenv)
    items, knapsacks, _, _ = env.observe()
    _items = torch.from_numpy(items).float()
    _knapsacks = torch.from_numpy(knapsacks).float()
    distil_score = model(_items, _knapsacks)
    gumbel_score = sample_gumbel(distil_score, sampling_number=sampling_number)
    gumbel_rank = get_rank(gumbel_score)    # [Num_sampling x num_items]
    return Gumbel_Search(items, knapsacks, gumbel_rank)
    # return 0


def calculate_zero_one(linear_score, rl_item_order, items, knapsacks):
    """
    :param linear_score: [1 x num_items]
    :param rl_item_order: [1 x num_items]
    :param items: [num_items x item_dim + 1]        # +1 for value
    :param knapsacks: [1 x item_dim]        # Just capacity
    """
    rl_zero_one = (rl_item_order != 0).int()
    linear_zero_one = torch.zeros_like(rl_zero_one)

    linear_score = linear_score.to("cpu")
    linear_softrank = soft_rank(linear_score, direction="DESCENDING", regularization_strength=5)
    linear_order = torch.argsort(linear_score, -1, descending=True).squeeze()

    cumulated_weights = torch.zeros_like(knapsacks)
    # cumulated_values = 0
    for i in range(linear_score.size(1)):       # i ~ num_items
        if (torch.sum(
            cumulated_weights + items[linear_order[i]][1:].unsqueeze(0) > knapsacks
        )):
            break
        linear_zero_one[0][linear_order[i]] = 1
        cumulated_weights += items[linear_order[i]][1:].unsqueeze(0)
        # cumulated_values += items[linear_order[i]][0].item()

    ret = torch.zeros_like(rl_zero_one)
    for i in range(ret.size(1)):        # i ~ num_items
        if rl_zero_one[0][i] == linear_zero_one[0][i]:
            continue
        elif rl_zero_one[0][i] > linear_zero_one[0][i]:
            ret[0][i] = 1
        else:
            ret[0][i] = -1
    ret_loss = torch.sum(ret * linear_softrank)
    return ret_loss

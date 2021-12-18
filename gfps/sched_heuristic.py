import numpy as np
import torch
import sched
import argparse
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
EPS = 1e-5
ZETA = 0.3


def scores_to_priority(scores):
    rank = np.argsort(-scores)
    priority = np.zeros_like(rank)
    for i in range(len(priority)):
        priority[rank[i]] = i
    return priority


def liu_score(tasks, num_procs=1):
    try:
        tasks = tasks.numpy()
    except:
        tasks = np.asarray(tasks)
    n = tasks.shape[0]

    left = (tasks[:, 1] / tasks[:, 0]).sum()

    right = num_procs * (n * (2 ** (1 / n) - 1))
    return left, right


def liu_test(tasks, num_procs=1):
    left, right = liu_score(tasks, num_procs)
    return int(left < right)


# 얘를 구현하고, 얘를 바까서 딱히 별 일이 없도록 하자.
def seperate_taskset(taskset, use_deadline=True):
    # length : 4
    # taskset[:, 0] : period, T
    # taskset[:, 1] : execution time, C
    # taskset[:, 2] : deadline. deadline, D is in between [C, T], uniform random integer.
    # 혹시 몰라, S는 slack time임. T - C

    if isinstance(taskset, torch.Tensor):
        taskset = taskset.numpy()


    T, C, D = (taskset[:, 0],
               taskset[:, 1],
               taskset[:, 2])


    if use_deadline:
        return T, C, D
    else:
        #implicit case, T is equal to deadline
        return T, C, T

def new_OPA(tasks, num_procs, test, use_deadline=True):
    """Audsley's Optimality Priority Assignment Algorithm"""

    l = len(tasks)
    priority = np.ones(shape=(l, ), dtype=np.int64) * l
    unassigned = set(range(l))

    for k in range(l):
        allocated = False
        for task in unassigned:
            prev_priority = priority[task]
            priority[task] = k
            res = test(tasks, num_procs, priority, use_deadline)
            if res is True:  # Schedulable
                unassigned.remove(task)
                allocated = True
                break
            else:
                priority[task] = prev_priority
        if allocated is False:
            return False, np.array(priority)

    return True, np.array(priority)


def OPA(tasks, num_procs, test, use_deadline=True):
    """Audsley's Optimality Priority Assignment Algorithm"""

    l = len(tasks)
    priority = np.ones(shape=(l, ), dtype=np.int64) * l
    unassigned = set(range(l))

    for k in range(l):
        allocated = False
        for task in unassigned:
            priority[task] = k
            res = test(tasks, num_procs, priority, use_deadline)
            if res is True:  # Schedulable
                unassigned.remove(task)
                allocated = True
                break
            else:
                priority[task] = l

        if allocated is False:
            return False, np.array(priority)

    return True, np.array(priority)

## Test_*_scores 함수에서는
## k가 높을 수록 priority가 높은 것 (논문 수식이 다 이렇게 전개되어 있어서 'ㅅ'...)
## k, k-1, ... 해서 0까지 integer 값임.

def test_C_RTA(tasks, num_procs, priority, use_deadline=True):
    #여기 들어오는 Priority는 Strict해야 한다.
    #0, 1, ..., l - 1 (l - 1)이 가장 높은 priority
    num_tasks, inventory = tasks.shape
    T, C, D = seperate_taskset(tasks, use_deadline)
    l = len(tasks)
    m = num_procs
    R_UB = np.copy(C)

    def W_R(i, L):
        N_R = (L + C[i] - C[i]) // T[i]
        return N_R * C[i] + min(C[i], L + C[i] - C[i] - N_R * T[i])

    def I_R(i, k, R_UB_k):
        return min(W_R(i, R_UB_k), R_UB_k - C[k] + 1)

    def W_NC(i, L):
        N_NC = (L // T[i])
        return N_NC * C[i] + min(C[i], L - N_NC * T[i])

    def I_NC(i, L, C_k):
        return min(W_NC(i, L), L - C_k + 1)

    def I_DIFF_R(i, k, R_UB_k):
        return I_R(i, k, R_UB_k) - I_NC(i, R_UB_k, C[k])

    def update_R(k, prev_R_k):
        left = 0
        for i in range(num_tasks):
            if priority[i] > priority[k]:
                left += I_R(i, k, prev_R_k)
        return C[k] + (left) // m

    def R(k):
        prev = R_UB[k] = C[k]
        for _ in range(100):
            R_UB[k] = update_R(k, R_UB[k])
            if np.abs(R_UB[k] - prev) <= 1e-5:
                break
            prev = R_UB[k]

    for p in reversed(range(l)):
        for k in range(l):
            if priority[k] == p:
                R(k)
            if D[k] < R_UB[k]:
                return False
    return True


def test_RTA(tasks, num_procs, priority, use_deadline=True, ret_score=False):
    #여기 들어오는 Priority는 Strict해야 한다.
    #0, 1, ..., l - 1 (l - 1)이 가장 높은 priority
    # 그리고 뭔가 Overflow가 남
    num_tasks, inventory = tasks.shape
    T, C, D = seperate_taskset(tasks, use_deadline)
    l = len(tasks)
    m = num_procs
    R_UB = np.copy(C)



    def W_R(i, L):
        N_R = (L + R_UB[i] - C[i]) // T[i]
        return N_R * C[i] + min(C[i], L + R_UB[i] - C[i] - N_R * T[i])

    def I_R(i, k, R_UB_k):
        ret = min(W_R(i, R_UB_k), R_UB_k - C[k] + 1)
        return ret

    def update_R(k, prev_R_k):
        left = 0
        for i in range(num_tasks):
            if priority[i] > priority[k]:
                left += I_R(i, k, prev_R_k)
        return C[k] + (left) // m

    def R(k):
        prev = R_UB[k]
        for _ in range(50):
            R_UB[k] = update_R(k, R_UB[k])
            if (np.abs(R_UB[k] - prev) <= 1e-5) or (D[k] < R_UB[k]):
                break
            prev = R_UB[k]

    if ret_score is True:
        ret = 0
        rett = True
        calcd = np.zeros(l, dtype=np.int32)
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    R(k)
                    if D[k] < R_UB[k]:
                        ret -= 1
                    else:
                        ret += 1
        if ret == l:
            return 2
        else:
            return (ret / l)
    else:
        ret = 0
        rett = True
        calcd = np.zeros(l, dtype=np.int32)
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    R(k)
                if D[k] < R_UB[k]:
                    return False
        return True


def test_RTA_LC(tasks, num_procs, priority, use_deadline=False, ret_score=False):
    #여기 들어오는 Priority는 Strict해야 한다.
    #0, 1, ..., l - 1 (l - 1)이 가장 높은 priority
    num_tasks, inventory = tasks.shape
    T, C, D = seperate_taskset(tasks, use_deadline)
    l = len(tasks)
    m = num_procs
    R_UB = np.copy(C)

    def W_R(i, L):
        N_R_ = (L + R_UB[i] - C[i]) // T[i]
        return N_R_ * C[i] + min(C[i], L + R_UB[i] - C[i] - N_R_ * T[i])

    def I_R(i, k, R_UB_k):
        return min(W_R(i, R_UB_k), R_UB_k - C[k] + 1)

    def W_NC(i, L):
        N_NC_ = (L // T[i])
        return N_NC_ * C[i] + min(C[i], L - N_NC_ * T[i])

    def I_NC(i, k, L):
        ret = min(W_NC(i, L), L - C[k] + 1)
        return ret

    def I_DIFF_R(i, k, R_UB_k):
        return I_R(i, k, R_UB_k) - I_NC(i, k, R_UB_k)

    def update_R(k, prev_R_k):
        left = 0
        for i in range(num_tasks):
            if priority[i] > priority[k]:
                left += I_NC(i, k, prev_R_k)

        right = []
        for i in range(num_tasks):
            if priority[i] > priority[k]:
                right.append(I_DIFF_R(i, k, prev_R_k))
        right = sorted(right, key=lambda x: -x)
        right = right[:m - 1]
        right = np.sum(right)
        return C[k] + (left + right) // m

    def R(k):
        prev = R_UB[k]
        for _ in range(100):
            R_UB[k] = update_R(k, R_UB[k])
            if (np.abs(R_UB[k] - prev) <= 1e-5) or (D[k] < R_UB[k]):
                break
            prev = R_UB[k]
    if ret_score is False:
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    R(k)
                    if D[k] < R_UB[k]:
                        return False
        return True
    else:
        ret = 0
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    R(k)
                    if D[k] < R_UB[k]:
                        ret += -1
                    else:
                        ret += 1
        if ret == l:
            return 2
        else:
            return (ret / l)
        return True


def test_DA(tasks, num_procs, priority, use_deadline, ret_score=False):
    #여기 들어오는 Priority는 Strict해야 한다.
    #0, 1, ..., l - 1 (l - 1)이 가장 높은 priority

    num_tasks, inventory = tasks.shape
    T, C, D = seperate_taskset(tasks, use_deadline)

    l = len(tasks)
    m = num_procs

    def N(i, L):
        return np.floor((L + D[i] - C[i]) / T[i])

    def W(i, L):
        return N(i, L) * C[i] + min(C[i], L + D[i] - C[i] - N(i, L) * T[i])

    def I(i, k):
        return min(W(i, D[k]), D[k] - C[k] + 1)
    if ret_score is False:
        for k in range(l):
            s = 0.0
            for i in range(l):
                if priority[i] > priority[k]:
                    s += I(i, k)
            r = C[k] + np.floor(s / m)
            if D[k] < r:
                return False
        return True
    else:
        ret = 0
        rett = True
        rs = []
        for k in range(l):
                    s = 0.0
                    for i in range(l):
                        if priority[i] > priority[k]:
                            s += I(i, k)
                    r = C[k] + np.floor(s / m)
                    rs.append(r - D[k])
                    if D[k] < r:
                        ret -= 1
                        rett = False
                    else:
                        ret += 1

        if ret == l:
            return 2
        else:
            return (ret / l)


        #return np.min(rs) / l
        #return (np.min(rs) - np.max(rs)) / l


def test_DA_LC(tasks, num_procs, priority, use_deadline, ret_score=False):
    #여기 들어오는 Priority는 Strict해야 한다.
    #0, 1, ..., l - 1 (l - 1)이 가장 높은 priority
    num_tasks, inventory = tasks.shape
    T, C, D = seperate_taskset(tasks, use_deadline)
    l = len(tasks)
    m = num_procs

    def N_D(i, L):
        return (L + D[i] - C[i]) // T[i]

    def W_D(i, L):
        ND = N_D(i, L)
        return ND * C[i] + min(C[i], L + D[i] - C[i] - ND * T[i])

    def N_NC(i, L):
        ret = (L // T[i])
        return ret

    def W_NC(i, L):
        return N_NC(i, L) * C[i] + min(C[i], L - N_NC(i, L) * T[i])

    def I_D(i, k):
        return min(W_D(i, D[k]), D[k] - C[k] + 1)

    def I_NC(i, k, L):
        ret = min(W_NC(i, L), L - C[k] + 1)
        return ret

    def I_DIFF_D(i, k, D_k):
        return I_D(i, k) - I_NC(i, k, D_k)

    def V(k):
        left = 0
        for i in range(num_tasks):
            if priority[i] > priority[k]:
                left += I_NC(i, k, D[k])

        right = []
        for i in range(num_tasks):
            if priority[i] > priority[k]:
                right.append(I_DIFF_D(i, k, D[k]))
        right = sorted(right, key=lambda x: -x)
        right = right[:m - 1]
        right = np.sum(right)
        return C[k] + (left + right) // m


    if ret_score is False:
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    if D[k] < V(k):
                        return False
        return True
    else:
        ret = 0
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    if D[k] < V(k):
                        ret += -1
                    else:
                        ret += 1
        if ret == l:
            return 2
        else:
            return (ret / l)
        return True


def test_NP_FP(tasks, num_procs, priority, use_deadline, ret_score=False):
    num_tasks, inventory = tasks.shape
    m = num_procs
    n = num_tasks
    T, C, D = seperate_taskset(tasks, use_deadline)
    S = D - C
    U = C / T
    U_tau = np.sum(U)

    I_1 = np.zeros(shape=(n, n), dtpye=np.int32)
    I_2 = np.zeros(shape=(n, n), dtpye=np.int32)
    I_df = np.zeros(shape=(n, n), dtpye=np.int32)

    def calc_I1(i, k):
        if i == k:
            return I1_1(k)

        if (priority[i] < priority[k]):
            if (A[k] == 0):
                return 0
            elif (alpha_2 >= A[k]) and (A[k] > 0):
                return I2_1[i]
            else:
                return I3_1[i]
        elif i == k:
            return I1_1(i)
        return I3_1[i];

    def calc_i2(i, k):
        if i == k:
            return I1_2[i]
        if (priority[i] < priority[k]) and (S[k] >= C[i]):
            return I3_2[i]
        return I4_2[i]


    for i in range(n):
        for k in range(n):
            I_1[i, k] = 0

    #right = sorted(right, key=lambda x: -x)
    #right = right[:m - 1]

def get_DM_DS_scores(tasks, num_procs, zeta=ZETA, use_deadline=True):
    T, C, D = seperate_taskset(tasks,  use_deadline=use_deadline)
    delta = np.array(C / D)
    arr = np.argsort(-delta)[:num_procs - 1]
    arr = np.array([delta[x] >= ZETA for x in arr.tolist()])
    order = np.copy(D)
    order[delta >= zeta] = -1
    return (order - C * EPS)

def get_SM_DS_scores(tasks, num_proces, zeta=ZETA, use_deadline=True):
    T, C, D = seperate_taskset(tasks, use_deadline=use_deadline)
    S = T - C
    delta = np.array(C / D)
    arr = np.argsort(-delta)[:num_procs - 1]
    arr = np.array([delta[x] >= ZETA for x in arr.tolist()])
    order = np.copy(S)
    order[delta >= zeta] = -1
    return (order - C * EPS)

def get_DkC_scores(tasks, num_procs=1, k=-1, use_deadline=True):
    #DkC라고 불리고, DCMPO라고도 불림;
    T, C, D = seperate_taskset(tasks, use_deadline=use_deadline)
    m = num_procs

    if k < 0:
        k = m - 1 + np.sqrt(5 * (m ** 2) - 6 * m + 1 + EPS)
        k /= (2 * m)

    order = D - C * k
    ret = np.zeros_like(order)
    for rank, idx in enumerate(np.argsort(order)):
        ret[idx] = rank
    return order
    return ret

def get_TkC_scores(tasks, num_procs, k=-1, use_deadline=True):
    #TkC라고 불리고, TCMPO라고도 불림.
    T, C, D = seperate_taskset(tasks)
    m = num_procs

    if k < 0:
        k = m - 1 + np.sqrt(5 * (m ** 2) - 6 * m + 1 + EPS)
        k /= (2 * m)

    order = T - C * k
    return order

def get_DCMPO_scores(tasks, num_procs, k=EPS, use_deadline=True):
    return get_DkC_scores(tasks, num_procs, k=k, use_deadline=use_deadline)

def get_DM_scores(tasks, num_procs, use_deadline=True):
    return get_DkC_scores(tasks, num_procs=num_procs, k=EPS, use_deadline=use_deadline)

def get_SM_scores(tasks, num_procs):
    return get_SM_US_scores(tasks, num_procs, zeta=10000.0)


num_procs = 16
sample_size = 50
pointset_size = 80

def check(tasks, order, num_procs):
    if isinstance(tasks, torch.Tensor):
        tasks = tasks.numpy()
    ret = sched.ScdChecker(tasks, order, num_procs=num_procs).run() > 0
    return ret

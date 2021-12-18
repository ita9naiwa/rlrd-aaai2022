import numpy as np
import cython
#cimport numpy as np
from libc.string cimport memset
from libcpp.unordered_set cimport unordered_set
EPS = 1e-5
ZETA = 0.3

def seperate_taskset(taskset, use_deadline=False):
    T, C, D = (taskset[:, 0],
               taskset[:, 1],
               taskset[:, 2])
    if use_deadline:
        return T, C, D
    else:
        #implicit case, T is equal to deadline
        return T, C, T


def OPA(tasks, num_procs, test, use_deadline=False):
    return OPA_DA_LC(tasks, use_deadline, num_procs, tasks.shape[0])

#def OPA_SCORE(tasks, num_procs, use_deadline=True):
#    return OPA_DA_LC_SCORE(tasks, use_deadline, num_procs, tasks.shape[0])

def OPACRTA(tasks, num_procs, test, use_deadline=False):
    return OPA_C_RTA(tasks, use_deadline, num_procs, tasks.shape[0])


@cython.boundscheck(False)
def OPA_C_RTA(tasks, use_deadline, long num_proc, long num_tasks):

    T, C, D = seperate_taskset(tasks, use_deadline)
    cdef long i, j, k
    cdef unordered_set[long] unassigned
    priority = np.array([num_tasks - 1 for _ in range(num_tasks)])
    cdef long allocated
    cdef long prev_priority

    for i in range(num_tasks):
        unassigned.insert(i)

    for k in range(num_tasks):
        allocated = 0
        for task in unassigned:
            prev_priority = priority[task]
            priority[task] = k
            res = _RTA_C(T, C, D, num_proc, num_tasks, priority)
            if res is True:
                unassigned.erase(task)
                allocated = 1
                break
            else:
                priority[task] = prev_priority
        if allocated == 0:
            return False
    return True

@cython.boundscheck(False)
def OPA_DA_LC(tasks, use_deadline, long num_proc, long num_tasks):
    T, C, D = seperate_taskset(tasks, use_deadline)
    if not use_deadline:
        D = T
    cdef long i, j, k
    cdef unordered_set[long] unassigned
    priority = np.array([num_tasks - 1 for _ in range(num_tasks)])
    cdef long allocated
    cdef long prev_priority

    for i in range(num_tasks):
        unassigned.insert(i)

    for k in range(num_tasks):
        allocated = 0
        for task in unassigned:
            prev_priority = priority[task]
            priority[task] = k
            res = _DA_LC(T, C, D, num_proc, num_tasks, priority)
            if res is True:
                unassigned.erase(task)
                allocated = 1
                break
            else:
                priority[task] = prev_priority
        if allocated == 0:
            return False
    return True

def test_DA_LC(tasks, num_proc, priorites, use_deadline, ret_score=0):
    T, C, D = seperate_taskset(tasks, use_deadline)
    num_tasks = T.shape[0]
    return _DA_LC(T, C, D, num_proc, num_tasks, priorites, ret_score)


@cython.boundscheck(False)
def OPA_DA_LC(tasks, use_deadline, long num_proc, long num_tasks):
    T, C, D = seperate_taskset(tasks, use_deadline)
    cdef long i, j, k
    cdef unordered_set[long] unassigned
    priority = np.array([num_tasks - 1 for _ in range(num_tasks)])
    cdef long allocated
    cdef long prev_priority

    for i in range(num_tasks):
        unassigned.insert(i)

    for k in range(num_tasks):
        allocated = 0
        for task in unassigned:
            prev_priority = priority[task]
            priority[task] = k
            res = _DA_LC(T, C, D, num_proc, num_tasks, priority)
            if res is True:
                unassigned.erase(task)
                allocated = 1
                break
            else:
                priority[task] = prev_priority
        if allocated == 0:
            return False
    return True


@cython.cdivision(True)
@cython.boundscheck(False)
def _DA_LC(long[:] T, long[:] C, long[:] D, long m, long l, long[:] priority):
    cdef int i, j, k, p
    def W_D(long i, L):
        ND = (L + D[i] - C[i]) // T[i]
        return ND * C[i] + min(C[i], L + D[i] - C[i] - ND * T[i])

    def W_NC(long i, L):
        N_NC = (L // T[i])
        return N_NC * C[i] + min(C[i], L - N_NC * T[i])

    def I_D(long i, long k):
        return min(W_D(i, D[k]), D[k] - C[k] + 1)

    def I_NC(long i, long k, L):
        ret = min(W_NC(i, L), L - C[k] + 1)
        return ret

    def I_DIFF_D(long i, long k, D_k):
        return I_D(i, k) - I_NC(i, k, D_k)

    def V(k):
        left = 0
        for i in range(l):
            if priority[i] > priority[k]:
                left += I_NC(i, k, D[k])
        right = []
        for i in range(l):
            if priority[i] > priority[k]:
                right.append(I_D(i, k) - I_NC(i, k, D[k]))
        right = sorted(right, key=lambda x: -x)
        right = right[:m - 1]
        right = np.sum(right)
        return C[k] + (left + right) // m

    for p in reversed(range(l)):
        for k in range(l):
            if priority[k] == p:
                if D[k] < V(k):
                    return False
    return True

def test_RTA_LC(tasks, num_proc, priorites, use_deadline, ret_score=0):
    T, C, D = seperate_taskset(tasks, use_deadline)
    num_tasks = T.shape[0]
    return _test_RTA_LC(T, C, D, num_proc, num_tasks, priorites, ret_score)

@cython.cdivision(True)
@cython.boundscheck(False)
def _test_RTA_LC(long[:] T, long[:] C, long[:] D, long m, long l, long[:] priority, long ret_score=0):
    cdef long[:] R_UB = np.copy(C)
    cdef int idx = 0
    cdef int i, j, k, p
    cdef float rew
    def W_R(long i, L):
        N_R_ = (L + R_UB[i] - C[i]) // T[i]
        return N_R_ * C[i] + min(C[i], L + R_UB[i] - C[i] - N_R_ * T[i])

    def I_R(long i, long k, R_UB_k):
        return min(W_R(i, R_UB_k), R_UB_k - C[k] + 1)

    def W_NC(long i, L):
        N_NC_ = (L // T[i])
        return N_NC_ * C[i] + min(C[i], L - N_NC_ * T[i])

    def I_NC(long i, long k, L):
        ret = min(W_NC(i, L), L - C[k] + 1)
        return ret

    def I_DIFF_R(long i, long k, R_UB_k):
        return I_R(i, k, R_UB_k) - I_NC(i, k, R_UB_k)

    def update_R(long k, prev_R_k):
        left = 0
        for i in range(l):
            if priority[i] > priority[k]:
                left += I_NC(i, k, prev_R_k)

        right = []
        for i in range(l):
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

    if ret_score == 0:
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    R(k)
                    if D[k] < R_UB[k]:
                        return 0
        return 1
    elif ret_score == 1:
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
    else:
        ret = np.zeros(shape=(l, ))
        violated = False
        rew = 1.0 / float(l)
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    R(k)
                    if D[k] < R_UB[k]:
                        violated = True
                        ret[idx] = -rew
                    else:
                        ret[idx] = rew
                    idx += 1

        if violated:
            return 0, ret
        else:
            return 1, ret


def test_Lee(tasks, num_proc, priorities, use_deadline, ret_score=0):
    T, C, D = seperate_taskset(tasks, use_deadline)
    num_tasks = T.shape[0]
    return _test_Lee(T, C, D, num_proc, num_tasks, priorities, ret_score)


@cython.cdivision(True)
@cython.boundscheck(False)
def _test_Lee(long[:] T, long[:] C, long[:] D, long m, long l, long[:] priority, long ret_score=0):
    cdef long[:] F_UB = np.ones_like(T)
    cdef int idx = 0
    cdef int i, j, k, p
    cdef float rew

    def W(i, l):
        NIL = (l + D[i] - C[i]) // T[i]
        return NIL * C[i] + min(C[i], l + D[i] - C[i] - NIL * T[i])

    def update_F(long k, long prev_F_k):
        left = 0
        right = []
        for i in range(l):
            if priority[i] > priority[k]:
                left += min(W(i, prev_F_k), prev_F_k)
            if priority[i] < priority[k]:
                right.append(min(C[i]-1, prev_F_k))

        right = sorted(right, key=lambda x: -x)
        right = right[:m]
        right = np.sum(right)
        return (left + right) // m

    def F(k):
        prev = 0
        for _ in range(100):
            F_UB[k] = update_F(k, F_UB[k])
            if (np.abs(F_UB[k] - prev) <= 1e-5) or F_UB[k] >= D[k] - C[k]:
                break
        prev = F_UB[k]
    if ret_score == 0:
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    F(k)
                    if F_UB[k] >= D[k] - C[k]:
                        return 0
        return 1
    elif ret_score == 2:
        ret = np.zeros(shape=(l, ))
        viloated = False
        rew = 1.0 / float(l)
        for p in reversed(range(l)):
            for k in range(l):
                if priority[k] == p:
                    F(k)
                    if F_UB[k] >= D[k] - C[k]:
                        violated = True
                        ret[idx] = -rew
                    else:
                        pass
                        ret[idx] = rew
                    idx += 1
                    break
        if violated is True:
            return 0, ret
        else:
            return 1, ret


@cython.cdivision(True)
@cython.boundscheck(False)
def _RTA_C(long[:] T, long[:] C, long[:] D, long m, long l, long[:] priority):
    cdef long[:] R_UB = np.copy(C)

    def W_R(long i, long L):
        N_R = L // T[i]
        return N_R * C[i] + min(C[i], L - N_R * T[i])

    def I_R(i, k, R_UB_k):
        ret = min(W_R(i, R_UB_k), R_UB_k - C[k] + 1)
        return ret

    def update_R(k, prev_R_k):
        left = 0
        for i in range(l):
            if priority[i] > priority[k]:
                left += I_R(i, k, prev_R_k)
        return C[k] + (left) // m

    def R(k):
        prev = -1
        for _ in range(50):
            R_UB[k] = update_R(k, R_UB[k])
            if (np.abs(R_UB[k] - prev) <= 1e-5) or (D[k] < R_UB[k]):
                break
            prev = R_UB[k]
        return R_UB[k]


    for p in reversed(range(l)):
        for k in range(l):
            if priority[k] == p:
                R(k)
            if D[k] < R_UB[k]:
                return False
    return True


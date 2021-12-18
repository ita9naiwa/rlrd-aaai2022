from tqdm import tqdm
import itertools
from queue import PriorityQueue
from functools import reduce
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import sched_heuristic as heu
from generator_emstada import StaffordRandFixedSum, gen_periods
R_FAILURE = -1
R_SUCCESS = 1
S_NOTINQUEUE = 0
S_INQUEUE = 1
S_RUNNING = 2
J_NOTRUNNING = -1
P_NOTHOLDING = -1

def liu_score(tasks):
    try:
        tasks = tasks.numpy()
    except:
        tasks = np.asarray(tasks)
    n = tasks.shape[0]

    left = (tasks[:, 1] / tasks[:, 0]).sum()

    right = n * (2 ** (1 / n) - 1)
    return left, right


class SchedT1Dataset(Dataset):
    # Util range can be in (0, num_procs)
    def __init__(self, num_procs, num_tasks=24, num_samples=1000,
                 period_range=(16, 256), util=0.5, gg=False):

        super(SchedT1Dataset, self).__init__()
        self.num_tasks = num_tasks
        self.num_procs = num_procs
        self.low, self.high = period_range
        self.util = util

        torch.manual_seed(1541)

        periods = gen_periods(self.num_tasks, num_samples, self.low, self.high, 1, dist="logunif").astype(np.int64)
        utils = StaffordRandFixedSum(self.num_tasks, util, num_samples)
        exec_time = np.floor(periods * utils).astype(np.int64)
        exec_time[exec_time < 1] = 1
        deadline = np.floor(np.random.uniform(exec_time, periods + 1, size=(num_samples, num_tasks))).astype(np.int64)
        self.data_set = np.stack([periods, exec_time, deadline], -1)
        print(self.data_set.shape)

    def setlen(self, newlen):
        self.data_set = self.data_set[:newlen]

    def __len__(self):
        return self.data_set.shape[0]

    def __getitem__(self, idx):
        return idx, self.data_set[idx]


class SchedT2Dataset(Dataset):
    # Util range can be in (0, num_procs)
    def __init__(self, num_procs, num_tasks=24, num_samples=1000,
                 period_range=(1, 5), util=0.5, gg=False):

        super(SchedT2Dataset, self).__init__()
        self.num_tasks = num_tasks
        self.num_procs = num_procs
        self.low, self.high = period_range
        self.util = util

        torch.manual_seed(1541)

        periods = gen_periods(self.num_tasks, num_samples, self.low, self.high, 1, dist="poweroftwo").astype(np.int64)
        utils = StaffordRandFixedSum(self.num_tasks, util, num_samples)
        exec_time = np.floor(periods * utils).astype(np.int64)
        exec_time[exec_time < 1] = 1
        deadline = np.floor(np.random.uniform(exec_time, periods + 1, size=(num_samples, num_tasks))).astype(np.int64)
        self.data_set = np.stack([periods, exec_time, deadline], -1)

    def setlen(self, newlen):
        self.data_set = self.data_set[:newlen]

    def __len__(self):
        return self.data_set.shape[0]

    def __getitem__(self, idx):
        return idx, self.data_set[idx]



def get_lcm(numbers):
    def gcd(x, y):
        if y:
            return gcd(y, x % y)
        else:
            return x

    def lcm(x, y):
        return (x * y) // gcd(x, y)

    return reduce(lcm, numbers)

def peak(pq):
  return pq.queue[0]


# Python Priority queue에 편리하게, 작은 수가 우선순위가 높게 구현되어 있음.
# Deadline 반영하게 수정될 필요가 있다.
class ScdChecker(object):
    def __init__(self, tasks, order_list, num_procs=1):
        """check schedulability of the tasks given the order.

        Parameters
        ----------
        tasks : list
            [num_tasks x 2 or 3 ] (period, execution time, DeadLine)
        order : list
            [num tasks] of arg list
        """

        self.num_procs = num_procs
        self.tasks = tasks
        self.order_list = order_list
        self.pq = PriorityQueue()
        self.eq = PriorityQueue()
        self.periods = periods = [task[0] for task in tasks]
        self.exec_times = [task[1] for task in tasks]
        try:
            self.deadlines = [task[2] for task in tasks]
        except:
            self.deadlines = self.periods

        self.hyperperiod = get_lcm(periods)
        self.running_job_cnt = 0
        self.processors = [P_NOTHOLDING for x in range(self.num_procs)]


    def run(self, ):
        def get_job_desc():
            return {"state": 0, "period": -1, "deadline": -1, "priority": -1, "time_left": -1, "exec_time": -1, "proc_idx": -1}


        self.job_set = [get_job_desc() for x in self.order_list]

        for i, (pe, pr, ex, dl) in enumerate(zip(self.periods, self.order_list, self.exec_times, self.deadlines)):
            self.job_set[i]["period"] = pe
            self.job_set[i]["priority"] = pr
            self.job_set[i]["exec_time"] = ex
            self.job_set[i]["deadline"] = dl

        self.cur_time = 0

        self.processors = [P_NOTHOLDING for x in range(self.num_procs)]

        for t in range(512):
            # 가장 먼저, 시간이 흘러야.
            running_procs = set()
            for proc_idx, proc in enumerate(self.processors):
                if proc == P_NOTHOLDING:
                    continue
                running_procs.add(proc_idx)

                if self.job_set[proc]["state"] != S_RUNNING:
                    raise IndexError("!Running job이 들어와 있음")
                running_job = proc
                if self.job_set[running_job]["state"] != S_RUNNING:
                    raise IndexError("...")
                self.job_set[running_job]["time_left"] -= 1
                # job이 끝나는 경우
                if self.job_set[running_job]["time_left"] == 0:
                    # 실행되던 job이 종료될 때
                    self.release_job(running_job)
                    running_procs.remove(proc_idx)

            for idx, p in enumerate(self.periods):
                if (t % p) == 0:
                    #Job arrival event
                    if self.job_set[idx]["state"] != S_NOTINQUEUE:
                        return R_FAILURE
                    self.job_set[idx]["state"] = S_INQUEUE
                    self.job_set[idx]["time_left"] = self.job_set[idx]["exec_time"]
                    #print(self.job_set[idx])
                    self.eq.put((self.job_set[idx]["priority"], idx))

            if self.eq.empty():
                continue
            while not self.eq.empty() and len(running_procs) < self.num_procs:
                for proc_idx in range(self.num_procs):
                    if proc_idx in running_procs:
                        continue
                    #아무것도 실행하고 있지 않은 processor
                    priority, job_idx = self.eq.get()
                    if self.job_set[job_idx]["state"] != S_INQUEUE:
                        raise IndentationError("'ㅅ'...")
                    self.allocate_job(job_idx, proc_idx)
                    running_procs.add(proc_idx)
                    break


            if self.eq.empty():
                continue

            if len(running_procs) == self.num_procs:
                nj_priority, idx = self.eq.get()
                success = self.try_preempt(nj_priority, idx)
                if success is False:
                    self.eq.put((nj_priority, idx))
            else:
                pass

        return R_SUCCESS

    def try_preempt(self, nj_priority, nj_idx):
        # 이게 실행될 때 모든 processor에 job이 일단은 들어가 있음.
        priorities_in_procs = [self.job_set[self.processors[proc_idx]]["priority"] for proc_idx in range(self.num_procs)]

        oj_proc_idx = np.argmax(priorities_in_procs)
        oj_job_idx = self.processors[oj_proc_idx]
        oj_priority = priorities_in_procs[oj_proc_idx]

        if oj_priority > nj_priority: #new job을 할당해야 함.
            self.job_set[oj_job_idx]["state"] = S_INQUEUE
            self.job_set[oj_job_idx]["proc_idx"] = -1
            self.eq.put((oj_priority, oj_job_idx))
            self.processors[oj_proc_idx] = P_NOTHOLDING
            self.allocate_job(nj_idx, oj_proc_idx)
            return True
        else:
            return False

    def release_job(self, job_idx):
        proc_idx = self.job_set[job_idx]["proc_idx"]

        if proc_idx == -1:
            raise IndexError("job은 어떤 proc에 할당되어 있어야 하는데")

        if self.job_set[job_idx]["time_left"] != 0:
            raise IndexError("왜 left time이 0이 아닌게 release됨?")
        self.job_set[job_idx]["state"] = S_NOTINQUEUE
        self.processors[proc_idx] = P_NOTHOLDING

    def allocate_job(self, job_idx, proc_idx):
        if self.processors[proc_idx] != P_NOTHOLDING:
            raise IndexError("잡이 실행되고 있지 않아야 allocation이 가능함")

        self.job_set[job_idx]["state"] = S_RUNNING
        self.job_set[job_idx]["proc_idx"] = proc_idx
        self.processors[proc_idx] = job_idx

from sklearn.utils import shuffle


if __name__ == "__main__":
    # Schedulability Test
    num_procs = 2
    n_tasks = 5
    for n_tasks in (2, 4, 6, 8, 10, 12, 14, 16):
        ds = SchedT1Dataset(2, n_tasks, 100, period_range=(1, 1000), util=1.3, gg=True)
        rret = []
        scrable = []
        for x, tasks in ds:
            tmp = []
            num_sc = 0
            num_not = 0
            for _ in range(10000):
                order_list = np.random.permutation(n_tasks)
                ret = heu.test_RTA_LC(tasks, num_procs, order_list, True, ret_score=0)
                if ret > 0:
                    num_sc += 1
                else:
                    num_not += 1
            r = num_sc / (num_sc+num_not)
            rret.append(r)
            scrable.append(num_sc > 0)
        print("at n_tasks:%d" % n_tasks, 100 *np.mean(rret), "percents orders out of all possible orders are schedulable", "sc percent", 100 * np.mean(scrable))
    """
    at n_tasks:2 100.0 percents orders out of all possible orders are schedulable sc percent 100.0
    at n_tasks:4 59.81859999999999 percents orders out of all possible orders are schedulable sc percent 100.0
    at n_tasks:6 39.50750000000001 percents orders out of all possible orders are schedulable sc percent 100.0
    at n_tasks:8 23.0785 percents orders out of all possible orders are schedulable sc percent 100.0
    at n_tasks:10 15.3839 percents orders out of all possible orders are schedulable sc percent 100.0
    at n_tasks:12 8.5116 percents orders out of all possible orders are schedulable sc percent 99.7
    at n_tasks:14 5.854699999999999 percents orders out of all possible orders are schedulable sc percent 99.2
    at n_tasks:16 3.0504 percents orders out of all possible orders are schedulable sc percent 98.6
    """

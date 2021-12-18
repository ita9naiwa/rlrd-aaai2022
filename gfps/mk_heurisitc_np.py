
import argparse
import numpy as np
import cy_heuristics as cy_heu
import sched_heuristic as heu
import pickle
from scipy.stats import kendalltau as tau, spearmanr as rho
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--num_proc", type=int, default=4)
parser.add_argument("--num_task", type=int, default=16)
parser.add_argument("--use_deadline", type=bool, default=False)

args = parser.parse_args()

num_proc = args.num_proc
use_deadline = args.use_deadline

print("num_proc=", num_proc)
print("use_deadline", use_deadline)

def scores_to_priority(scores):
    rank = np.argsort(-scores)
    priority = np.zeros_like(rank)
    for i in range(len(priority)):
        priority[rank[i]] = i
    return priority

def get_util_range(num_proc):
    util = [str(x) for x in range(10, num_proc * 100, 10)]
    ret = []
    for x in util:
        if len(x) == 2:
            ret.append('0.' + x)
        else:
            ret.append(x[:len(x) - 2] + '.' + x[len(x) - 2:])

    return ret



res_map = defaultdict(lambda: defaultdict(lambda: 0))
for num_tasks in [args.num_task]:
    if num_proc >= num_tasks:
        continue
    util_range = get_util_range(num_proc)
    for util in util_range:
        print(num_tasks, util)
        res = res_map[(num_proc, num_tasks, util)]
        with open("eval/%d-%d/%s" % (num_proc, num_tasks, util), 'rb') as f:
            train_dataset = pickle.load(f)
        i = 0
        for x, y in train_dataset:

            if i == 10000:
                break
            opa_da_lc_res_ = cy_heu.OPA(y, num_proc, heu.test_DA_LC, use_deadline)
            #res['OPA_DA'] += opa_da_res
            res['OPA_DA_LC'] += opa_da_lc_res_

            p = scores_to_priority(heu.get_DkC_scores(y, num_proc, use_deadline=use_deadline))
            res['DkC_RTA_LC'] += (cy_heu.test_Lee(y, num_proc, p, use_deadline))

            p = scores_to_priority(heu.get_DM_scores(y, num_proc, use_deadline=use_deadline))
            res['DM_RTA_LC'] += (cy_heu.test_Lee(y, num_proc, p, use_deadline))

            p = scores_to_priority(heu.get_DM_DS_scores(y, num_proc, use_deadline=use_deadline))
            res['DM_DS_RTA_LC'] += (cy_heu.test_Lee(y, num_proc, p, use_deadline))

            p = scores_to_priority(heu.get_SM_DS_scores(y, num_proc, use_deadline=use_deadline))
            res['SM_DS_RTA_LC'] += (cy_heu.test_Lee(y, num_proc, p, use_deadline))


            i += 1

        res_map[(num_proc, num_tasks, util)] = dict(res)
        if use_deadline:
            with open("repo/heuristics_result_%d_%d_dl.pkl" % (num_proc, num_tasks), 'wb') as ff:
                pickle.dump(dict(res_map), ff)
        else:
            with open("repo/heuristics_result_%d_%d.pkl" % (num_proc, num_tasks), 'wb') as ff:
                pickle.dump(dict(res_map), ff)

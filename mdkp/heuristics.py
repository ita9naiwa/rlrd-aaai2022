
# install via `pip install ortool`
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.algorithms import pywrapknapsack_solver
from engine import timer
import time
# SOLVER = "GLOP"


"""
list of solvers
solver_id is case insensitive, and the following names are supported:
  - CLP_LINEAR_PROGRAMMING or CLP
  - CBC_MIXED_INTEGER_PROGRAMMING or CBC
  - GLOP_LINEAR_PROGRAMMING or GLOP
  - BOP_INTEGER_PROGRAMMING or BOP
  - SAT_INTEGER_PROGRAMMING or SAT or CP_SAT
  - SCIP_MIXED_INTEGER_PROGRAMMING or SCIP
  - GUROBI_LINEAR_PROGRAMMING or GUROBI_LP
  - GUROBI_MIXED_INTEGER_PROGRAMMING or GUROBI or GUROBI_MIP
  - CPLEX_LINEAR_PROGRAMMING or CPLEX_LP
  - CPLEX_MIXED_INTEGER_PROGRAMMING or CPLEX or CPLEX_MIP
  - XPRESS_LINEAR_PROGRAMMING or XPRESS_LP
  - XPRESS_MIXED_INTEGER_PROGRAMMING or XPRESS or XPRESS_MIP
  - GLPK_LINEAR_PROGRAMMING or GLPK_LP
  - GLPK_MIXED_INTEGER_PROGRAMMING or GLPK or GLPK_MIP
"""


def transform(items, knapsack, item_amounts, numpy=True):
    """
    Data example
    인풋 넣으면 대강 이렇게 돌려줌
        {'weights': array([[44.0625, 58.875 , 55.125 ],
                [14.4375, 16.6875, 63.1875],
                [15.375 , 19.875 , 52.125 ],
                [53.625 , 56.0625, 23.625 ],
                [20.625 , 56.0625, 23.625 ],
                [29.8125, 11.25  , 41.4375],
                [17.8125, 20.625 , 14.4375],
                [28.6875, 36.5625, 43.125 ]]),
        'values': array([ 8., 10., 11.,  2., 12., 11.,  8., 10.]),
        'amounts': [5, 1, 4, 3, 2, 2, 5, 4],
        'items': [0, 1, 2, 3, 4, 5, 6, 7],
        'num_items': 8,
        'bin_capacities': array([213.79110544, 185.22000405, 205.46047226])}
    """
    data = {}
    values = np.array(items[:, 0], dtype=np.double)
    weights = items[:, 1:]
    if not numpy:
        return values.tolist(), weights.T.tolist(), knapsack[0]
    data['weights'] = weights
    data['values'] = values
    data['amounts'] = item_amounts
    data['items'] = list(range(len(weights)))
    data['num_items'] = len(weights)
    num_bins = len(knapsack)
    data["bins"] = list(range(num_bins))
    data['bin_capacities'] = knapsack
    return data


class Heuristic_Solver(object):
    def __init__(self, solver, time_limit=False):
        self._solver = pywraplp.Solver.CreateSolver(solver)

        if time_limit:
            self._solver.set_time_limit(time_limit)

    def solve(self, items, knapsack, item_amounts, get_value=True, getlen=False):
        solver = self._solver
        data = transform(items, knapsack, item_amounts)
        x = {}
        for i in data['items']:
            for j in data["bins"]:
                min_x = 0
                max_x = int(data['amounts'][i])
                x[(i, j)] = solver.IntVar(min_x, max_x, 'x_%d_%d' % (i, j))

        num_dims = data["weights"].shape[1]
        num_bins = len(data['bins'])

        for j in data["bins"]:
            for k in range(num_dims):
                solver.Add(
                    sum(x[(i, j)] * data['weights'][i][k] for i in data['items']) <= data['bin_capacities'][j, k])

        objective = solver.Objective()
        for i in data['items']:
            for j in data["bins"]:
                objective.SetCoefficient(x[(i, j)], data['values'][i])
        objective.SetMaximization()

        status = solver.Solve()
        status2 = pywraplp.Solver.OPTIMAL

        ret = []
        for j in range(num_bins):
            k_ret = np.array([int(x[(i, j)].solution_value()) for i in data['items']])
            ret.append(k_ret)

        ret = np.asarray(ret)
        _ret = np.dot(ret, data["weights"])
        assert np.all(_ret <= data["bin_capacities"] + 1)
        a = np.array(item_amounts)
        b = np.array(ret)
        assert np.all(a >= b)

        if not get_value:
            return status == pywraplp.Solver.OPTIMAL, ret
        elif get_value and not getlen:
            score = np.dot(ret, data['values']).sum()
            # if score.shape[0] == 0:
            #     score = score[0]
            return status, ret, score
        else:
            score = np.dot(ret, data['values']).sum()
            # if score.shape[0] == 0:
            #     score = score[0]
            return status, ret, score, np.sum(ret)


# class HeuristicSolver(object):
#     def __init__(self):
#         self.__solver__ = pywrapknapsack_solver.KnapsackSolver
#
#     def solve(self, items, knapsack, item_amounts, timelimit=False):
#         profit, weight, cap = transform(items, knapsack, item_amounts, numpy=False)
#         solver = self.__solver__(
#             pywrapknapsack_solver.KnapsackSolver
#             .KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
#             "Multi-dimensional solver"
#         )
#         solver.Init(profit, weight, cap)
#
#         if timelimit:
#             solver.set_time_limit(timelimit)
#
#         profit = solver.Solve()
#
#
#         return profit

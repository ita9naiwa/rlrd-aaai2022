"""
RL모델만 테스트.
"""
import torch

from MDKP_env import Env
from engine import simplest_greedy, test_rl
from heuristics import Heuristic_Solver
from hyperparams import *
from network import Att_Policy

NUM_TEST = 100

def main():
    if args.type == "gl":
        file_name = "../knapsackmodels/globalrl/1213/RL-it%d-dim%d-w%d-c%.2f" \
                    % (args.num_items, args.item_dim, args.w, args.c)
        # file_name = "../knapsackmodels/globalrl/RL-it200-dim20-w300-v100-a0.50"
    else:
        file_name = "../knapsackmodels/localrl/1213/LRL-it%d-dim%d-w%d-c%.2f" % (
            args.num_items, args.item_dim, args.w, args.c
        )
    MyModel = Att_Policy(args.item_dim, args.embed_dim, args.use_cuda)
    heuristic = Heuristic_Solver("GLOP")
    tmpheu = Heuristic_Solver("GLOP")

    with open(file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f)

    MyModel.load_state_dict(tmp.state_dict())
    MyModel.eval()

    env = Env(
        args.item_dim, args.num_items, 1,
        args.w, args.c
    )

    ratio = 0.0
    ratio2 = 0.0

    rlsum = 0
    heusum = 0
    greedysum = 0

    rllen = 0
    heulen = 0
    tmpheulen = 0
    greedylen = 0

    for i in range(NUM_TEST):
        observation, item_amounts = env.reset()
        MyModel.reset()
        items, knapsack, allocable_items,\
            allocable_knapsacks = observation
        status, coeff, heu_score, hlen = \
            heuristic.solve(items, knapsack, item_amounts, True, True)

        status2, coeff2, heu_score2, hlen2 = \
            tmpheu.solve(items, knapsack, item_amounts, True, True)
        tmpheulen += hlen2
        heulen += hlen
        heusum += heu_score

        _, greedy_score = simplest_greedy(items, knapsack)
        greedysum += greedy_score
        greedylen += _

        # while 1:
        #     items, knapsack, allocable_items, allocable_knapsacks =\
        #         observation
        #     items = torch.from_numpy(items).float()
        #     knapsack = torch.from_numpy(knapsack).float()
        #     item, log_prob = MyModel(
        #         items, knapsack, allocable_items,
        #         allocable_knapsacks, True)
        #     next_observation, reward, done = env.step(item, 0)
        #     rllen = rllen + 1
        #     if done:
        #         break
        #     observation = next_observation
        # dist_value = env.total_value()
        dist_value, _ = test_rl(env, MyModel)
        rlsum += dist_value
        ratio += dist_value / heu_score
        ratio2 += dist_value / greedy_score

        print("----------")
        print("OPTIMAL VALUE :", heu_score)
        print("GREEDY VALUE :", greedy_score)
        print("RL VALUE :", dist_value)
        print("RL / Heuristic :", dist_value / heu_score)
        print("RL / Greedy :", dist_value / greedy_score)
        print("----------")

    print("EPISODIC LEN, RL {}, SCIP {}, GLOP {}, GREEDY {}".format(rllen / NUM_TEST, heulen / NUM_TEST,
                                                                    tmpheulen / NUM_TEST, greedylen / NUM_TEST))

    print("FINAL RL : ", rlsum/NUM_TEST)
    print("FINAL Optimal : ", heusum/NUM_TEST)
    print("FINAL Greedy : ", greedysum/NUM_TEST)
    print("Final RL / Heu : ", ratio / NUM_TEST)
    print("Final RL / Greedy : ", ratio2 / NUM_TEST)


if __name__ == "__main__":
    main()

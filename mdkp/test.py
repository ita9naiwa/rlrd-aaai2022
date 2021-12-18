"""
distillation(+Gumbel), RL, Heuristics, Greedy 비교.
"""

import time

import torch

from MDKP_env import Env
from engine import test_rl, test_distil, simplest_greedy, test_gumbel
from heuristics import Heuristic_Solver
from hyperparams import *
from network import Encoder_Distilation, Att_Policy

NUM_TEST = 5
RL_SAMPLING = 1
GUMBEL_SAMPLING = 30

result_file_name = "it%d-dim%d-w%d-c%.2f" % (
    args.num_items, args.item_dim, args.w, args.c
)

if __name__ == "__main__":
    print("it%d-dim%d-w%d-c%.2f"%(
    args.num_items, args.item_dim, args.w, args.c
    ))
    # Load distillation model
    distillation_file_name = "DISTIL-it%d-dim%d-w%d-c%.2f" \
          % (
            args.num_items, args.item_dim, args.w, args.c
          )
    reinforce_file_name = "LRL-it%d-dim%d-w%d-c%.2f" % (
        args.num_items, args.item_dim, args.w, args.c
    )

    d_model = Encoder_Distilation(
        args.item_dim, args.embed_dim, args.use_cuda
    )

    with open("../mdkpmodels/distillation/" + distillation_file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f, map_location=torch.device("cpu"))
    d_model.load_state_dict(tmp.state_dict())

    # Load reinforcement model

    r_model = Att_Policy(args.item_dim, args.embed_dim, args.use_cuda)
    with open("../mdkpmodels/localrl/" + reinforce_file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f, map_location=torch.device("cpu"))
    r_model.load_state_dict(tmp.state_dict())

    # Define heuristics
    glop_heuristic = Heuristic_Solver("GLOP")

    # Define environment
    env = Env(
        args.item_dim, args.num_items, 1,
        args.w, args.c
    )

    glop_relapsed, srd_relapsed, greedy_relapsed, rl_relapsed, rls_relapsed, gu_relapsed = 0, 0, 0, 0, 0, 0
    glop, g, r, rs, d, gu = 0, 0, 0, 0, 0, 0

    # Test loop
    for i in range(NUM_TEST):
        print("%d th test"%(i))
        observation, item_amounts = env.reset()
        items, knapsacks, _, _ = observation

        # Heuristics (GLOP)
        glop_start = time.time()
        glopstatus, glopcoeff, glop_value = glop_heuristic.solve(
            items, knapsacks, item_amounts
        )
        glop_done = time.time()
        glop_relapsed += (glop_done - glop_start)
        glop += glop_value

        # Greedy
        greedy_start = time.time()
        greedy_num_of_items, greedy_value = simplest_greedy(items, knapsacks)
        greedy_end = time.time()
        greedy_relapsed += (greedy_end - greedy_start)
        g += greedy_value

        # RL
        rl_start = time.time()
        _rl_value, _ = test_rl(env, r_model)
        rl_end = time.time()
        rl_relapsed += (rl_end - rl_start)
        r += _rl_value

        # RL-Sampling
        rl_sampling_start = time.time()
        tmp = []
        for _ in range(RL_SAMPLING):
           rl_value, rl_num_of_items = test_rl(env, r_model, argmax=False)
           tmp.append(rl_value)
        rl_sampling_end = time.time()
        rls_relapsed += (rl_sampling_end - rl_sampling_start)
        rl_value = max(tmp)
        rs += rl_value

        # Distillation
        dist_start = time.time()
        distil_value = test_distil(env, d_model)
        dist_end = time.time()
        srd_relapsed += (dist_end - dist_start)
        d += distil_value

        # Distillation + Gumbel
        gum_start = time.time()
        gumbel_value = test_gumbel(env, d_model, GUMBEL_SAMPLING)
        gum_end = time.time()
        gu_relapsed += (gum_end - gum_start)
        gu += gumbel_value

    print("\n\n")

    print("**Average\n")
    print("GLOP: ", int(glop / NUM_TEST))
    print("Greedy: ", int(g / NUM_TEST))
    print("RL: ", int(r / NUM_TEST))
    print("RL(S): ", int(rs / NUM_TEST))
    print("Distillation: ", int(d / NUM_TEST))
    print("Gumbel: ", int(gu / NUM_TEST))

    print("** Ratio with optimal")
    print("GLOP / heu", glop / glop * 100)
    print("Greedy / heu", g / glop * 100)
    print("RL / heu", r / glop * 100)
    print("RL(S) / heu", rs / glop * 100)
    print("Distillation / heu", d / glop * 100)
    print("Gumbel / opt", gu / glop * 100)
    print("\n\n")
    print("GLOP    ", int(glop / NUM_TEST), glop/glop*100, glop_relapsed / NUM_TEST)
    
    print("Greedy  ", int(g/NUM_TEST), g/glop*100, greedy_relapsed / NUM_TEST)
    print("RL      ", int(r/NUM_TEST), r/glop*100, rl_relapsed / NUM_TEST)
    print("RL(S)   ", int(rs/NUM_TEST), rs/glop*100, rls_relapsed / NUM_TEST)
    print("SRD     ", int(d/NUM_TEST), d/glop*100, srd_relapsed / NUM_TEST)
    print("SRD-G   ", int(gu/NUM_TEST), gu/glop*100, gu_relapsed / NUM_TEST)

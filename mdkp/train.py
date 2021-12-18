import os
import time
import torch.optim as optim
from engine import *


def main():
    directory = os.path.dirname("../knapsackmodels/globalrl/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = "RL-it%d-dim%d-w%d-c%.2f" \
          % (
            args.num_items, args.item_dim, args.w, args.c
          )

    env = Env(
        args.item_dim, args.num_items, 1,
        args.w, args.c
    )

    if args.use_cuda:
        policy = Att_Policy(args.item_dim, args.embed_dim, args.use_cuda).to("cuda:0")
        bl = Att_Policy(args.item_dim, args.embed_dim, args.use_cuda).to("cuda:0")
    else:
        policy = Att_Policy(args.item_dim, args.embed_dim, args.use_cuda)
        bl = Att_Policy(args.item_dim, args.embed_dim, args.use_cuda)

    if args.load:
        with open("../knapsackmodels/globalrl/" + file_name + ".torchmodel", "rb") as f:
            tmp = torch.load(f)
        policy.load_state_dict(tmp.state_dict())

    bl.load_state_dict(policy.state_dict())
    policy.train()

    optimizer = optim.Adam(policy.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    averaged_value = 0
    averaged_returns = 0.0
    averaged_loss = 0.0

    start = time.time()
    for epoch in range(args.epochs):
        if epoch < 100:
            guide = True
        else:
            guide = False
        optimizer.zero_grad()
        loss, value, reward = rollout(env, policy, guide=guide)
        optimizer.step()
        scheduler.step()

        if epoch == 0:
            averaged_loss = loss
            averaged_value = value
            averaged_returns = reward

        else:
            averaged_returns = 0.95 * averaged_returns + 0.05 * reward
            averaged_value = 0.95 * averaged_value + 0.05 * value
            averaged_loss = 0.95 * averaged_loss + 0.05 * loss

        # early_stopping(-averaged_returns, policy)
        end = time.time()
        elapsed = end - start
        minute = int(elapsed//60)
        second = int(elapsed - 60 * minute)
        print("----------")
        print("EPOCH : {} / {}".format(epoch, args.epochs))
        print("CURRENT VALUE : {}".format(value))
        print("LOSS : {:.3f}, VALUE : {}, REWARD : {:.3f}".\
              format(averaged_loss, averaged_value, averaged_returns))
        print("ELAPSED TIME : {}m{}s".format(minute, second))

        # with open("log/globalrl/" + file_name, "w") as f:
        #     print("EPOCH : {}, VALUE : {}".format(epoch, value), file=f)

        # if early_stopping.early_stop:
        #     exit(0)


        torch.save(policy, "../knapsackmodels/globalrl/" + file_name + ".torchmodel")


if __name__ == "__main__":
    main()

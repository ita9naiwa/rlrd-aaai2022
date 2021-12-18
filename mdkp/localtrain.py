import os
import torch.optim as optim
from engine import *


def main():
    directory = os.path.dirname("../mdkpmodels/localrl/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    global_name = "RL-it200-dim20-w200-c0.00"
    # c = 0 --> Uncorrelated
    # c = 1 --> Correlated
    file_name = "LRL-it%d-dim%d-w%d-c%.2f" % (
        args.num_items, args.item_dim, args.w,  args.c
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

    policy.load_finetune_checkpoint(
        "../mdkpmodels/globalrl/"+global_name+".torchmodel"
    )

    if args.load:
        with open("../mdkpmodels/localrl/" + file_name + ".torchmodel", "rb") as f:
            tmp = torch.load(f)
        policy.load_state_dict(tmp.state_dict())

    for name, param in policy.named_parameters():
        if not name.startswith("item"):
            param.requires_grad = False
    bl.load_state_dict(policy.state_dict())
    policy.train()

    optimizer = optim.Adam(policy.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    averaged_value = 0
    averaged_returns = 0.0
    averaged_loss = 0.0

    start = time.time()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss, value, reward = rollout(env, policy, guide=True)
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

        end = time.time()
        elapsed = (end - start)
        print("----------")
        print("EPOCH : {} / {}".format(epoch, args.epochs))
        print("CURRENT VALUE : {}".format(value))
        print("LOSS : {:.3f}, VALUE : {}, REWARD : {:.3f}".\
              format(averaged_loss, averaged_value, averaged_returns))
        # print(elapsed)
        minute = int(elapsed // 60)
        second = int(elapsed - 60 * minute)
        print("경과시간 : {}m{:}s".format(minute, second))

        torch.save(policy, "../mdkpmodels/localrl/" + file_name + ".torchmodel")


if __name__ == "__main__":
    main()

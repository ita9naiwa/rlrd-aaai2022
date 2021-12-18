import os
import torch
import torch.nn as nn
import torch.optim as optim
from MDKP_env import Env
from network import Att_Policy, Encoder_Distilation
from hyperparams import *
from engine import train_distilation

ZERO_ONE_LOSS = True

def main():
    directory = os.path.dirname("../knapsackmodels/distillation/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    saving_file_name \
        = "DISTIL-it%d-dim%d-w%d-c%.2f" \
          % (
            args.num_items, args.item_dim, args.w, args.c
          )
    file_name = "LRL-it%d-dim%d-w%d-c%.2f" % (
        args.num_items, args.item_dim, args.w, args.c
    )

    Teacher = Att_Policy(
        args.item_dim, args.embed_dim, args.use_cuda,
    )

    with open("../knapsackmodels/localrl/"
              + file_name
              + ".torchmodel", "rb") as f:
        tmp = torch.load(f)

    Teacher.load_state_dict(tmp.state_dict())

    Student = Encoder_Distilation(
        args.item_dim, args.embed_dim, args.use_cuda
    )
    Student.load_state_dict(Teacher.state_dict(), strict=False)

    if args.load:
        with open("../knapsackmodels/distillation/"
                  + saving_file_name + ".torchmodel", "rb") as f:
            tmp = torch.load(f)
        Student.load_state_dict(tmp.state_dict())

    if args.use_cuda:
        Teacher = Teacher.to("cuda:0")
        Student = Student.to("cuda:0")
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(Student.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.85)

    env = Env(
        args.item_dim, args.num_items, 1,
        args.w, args.c
    )

    avg_loss = 0.0
    epoch_loss = 0.0
    for epoch in range(10000):
        loss = train_distilation(env, Teacher, Student, criterion, zeroone_loss=ZERO_ONE_LOSS)
        epoch_loss += loss
        if epoch % 16 == 0:
            epoch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            epoch_loss = 0.0
        if avg_loss <= 0:
            avg_loss = loss
        else:
            avg_loss = 0.95 * avg_loss + 0.05 * loss
        print(avg_loss)
        if epoch % 100 == 0:
            torch.save(Student, f="../knapsackmodels/distillation/" + saving_file_name + ".torchmodel")


if __name__ == "__main__":
    main()

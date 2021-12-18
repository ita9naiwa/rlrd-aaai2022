import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--type", choices=["gl", "lo"])
parser.add_argument("--global", action="store_true")
parser.add_argument("--num_knapsacks", type=int, default=1)
parser.add_argument("--num_items", type=int, default=15)
parser.add_argument("--item_dim", type=int, default=10)
parser.add_argument("--w", type=int)
parser.add_argument("--c", type=float, choices=[0, 0.9])
parser.add_argument("--discount_factor", type=float, default=1)
parser.add_argument("--embed_dim", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--clipping_const", type=float, default=10)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=250)
parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--prelen", type=int, default=10)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--load", action="store_true")

args = parser.parse_args()

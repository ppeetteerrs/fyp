from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

train_parser = ArgumentParser(description="StyleGAN2 trainer")

train_parser.add_argument("path", type=str, help="path to the lmdb dataset")

train_parser.add_argument(
    "--iter", type=int, default=200, help="total training iterations"
)

train_parser.add_argument(
    "--batch", type=int, default=8, help="batch sizes for each gpus"
)

train_parser.add_argument(
    "--n_sample",
    type=int,
    default=32,
    help="number of the samples generated during training",
)

train_parser.add_argument(
    "--size", type=int, default=1024, help="image sizes for the model"
)

train_parser.add_argument(
    "--r1", type=float, default=10, help="weight of the r1 regularization"
)

train_parser.add_argument(
    "--path_regularize",
    type=float,
    default=2,
    help="weight of the path length regularization",
)

train_parser.add_argument(
    "--path_batch_shrink",
    type=int,
    default=2,
    help="batch size reducing factor for the path length regularization (reduce memory consumption)",
)

train_parser.add_argument(
    "--d_reg_every",
    type=int,
    default=16,
    help="interval of the applying r1 regularization",
)

train_parser.add_argument(
    "--g_reg_every",
    type=int,
    default=4,
    help="interval of the applying path length regularization",
)

train_parser.add_argument(
    "--mixing", type=float, default=0.9, help="probability of latent code mixing"
)

train_parser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help="path to the checkpoints to resume training",
)

train_parser.add_argument("--lr", type=float, default=0.002, help="learning rate")

train_parser.add_argument(
    "--channel_multiplier",
    type=int,
    default=2,
    help="channel multiplier factor for the model. config-f = 2, else = 1",
)


class TrainArgs:
    def __init__(self, args: Namespace) -> None:
        self.path: str = args.path
        self.iter: int = args.iter
        self.batch: int = args.batch
        self.n_sample: int = args.n_sample
        self.size: int = args.size
        self.r1: float = args.r1
        self.path_regularize: float = args.path_regularize
        self.path_batch_shrink: int = args.path_batch_shrink
        self.d_reg_every: int = args.d_reg_every
        self.g_reg_every: int = args.g_reg_every
        self.mixing: float = args.mixing
        self.ckpt: Optional[str] = args.ckpt
        self.lr: float = args.lr
        self.channel_multiplier: int = args.channel_multiplier
        self.latent: int = 512
        self.n_mlp: int = 8
        self.start_iter: int = 0
        self.n_colors: int = 1
        self.device: str = "cuda"

        Path("results/sample").mkdir(parents=True, exist_ok=True)
        Path("results/checkpoint").mkdir(parents=True, exist_ok=True)


def get_train_args():
    return TrainArgs(train_parser.parse_args())

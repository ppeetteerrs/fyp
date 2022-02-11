from utils.config import CONFIG, guard

guard()

import os
from typing import Any, Dict, List, Tuple

import lpips
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeometry.losses import SSIM
from torchvision.transforms.functional import equalize
from torchvision.utils import save_image
from tqdm import tqdm

from psp.loss_fn.discriminator_loss import DiscriminatorLoss
from psp.loss_fn.id_loss import DiscriminatorIDLoss, IDLoss
from psp.loss_fn.reg_loss import RegLoss
from psp.pSp import pSp
from psp.ranger import Ranger
from utils.img import transform
from utils.lmdb import LMDBImageDataset, covid_ct_indexer
from utils.utils import repeat


class Coach:
    def __init__(self):
        self.net = pSp().to("cuda")

        # Initialize dataset
        self.test_dataset = LMDBImageDataset(
            CONFIG.COVID_19_TEST_LMDB,
            covid_ct_indexer,
            transform,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=5,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            drop_last=True,
        )

    def gen(self):
        self.net.eval()

        for idx, batch in enumerate(self.test_dataloader):
            raw, loc, drr, bone = [item.to("cuda") for item in batch]
            with torch.no_grad():

                img_in, img_out, img_style, w_plus = self.forward_pass((raw, loc))

            save_image(
                torch.concat(
                    [raw, img_out],
                    dim=3,
                ),
                str(CONFIG.OUTPUT_DIR / f"{str(idx).zfill(6)}.png"),
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

    def forward_pass(
        self, tensors: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Concat image along channel direction
        img_in = torch.cat([img for img in tensors], dim=1)

        img_out: Tensor
        img_style: Tensor
        w_plus: Tensor
        img_out, img_style, w_plus = self.net(img_in)
        return img_in, img_out, img_style, w_plus


def main():
    coach = Coach()
    coach.gen()


if __name__ == "__main__":
    main()

from utils.config import CONFIG, guard

guard()


import os
from typing import Dict, Literal, Tuple

import lpips
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeometry.losses import SSIM
from torchvision.utils import save_image
from tqdm import tqdm

from psp.loss_fn.discriminator_loss import (DenseDiscriminator,
                                            DiscriminatorLoss, IDLoss)
from psp.loss_fn.reg_loss import RegLoss
from psp.pSp_pure import pSp
from psp.ranger import Ranger
from utils.dataset import MulticlassImageDataset
from utils.utils import repeat, to_device


class Coach:
    def __init__(self):
        ckpt = torch.load(str(CONFIG.PSP_CKPT))
        self.net = pSp(ckpt).to("cuda")

        # Utility functions
        self.lpips_loss = lpips.LPIPS(net="vgg").to("cuda").eval()
        self.reg_loss = RegLoss().to("cuda").eval()
        self.id_loss = IDLoss().to("cuda").eval()
        self.discriminator = DenseDiscriminator().to("cuda").eval()
        self.discriminator_loss = DiscriminatorLoss().to("cuda").eval()
        self.ssim = SSIM(11, "mean", 2)

        # Initialize optimizer
        self.optimizer = Ranger(
            self.net.encoder.parameters(),
            lr=CONFIG.PSP_LR,
        )

        # Initialize dataset
        self.train_dataset = MulticlassImageDataset(
            [
                CONFIG.PROJECT_DIR / f"input/data/{item}/train"
                for item in CONFIG.PSP_DATASETS
            ],
            ["lung", "localizer", "bones", "drr", "soft"],
        )
        self.test_dataset = MulticlassImageDataset(
            [
                CONFIG.PROJECT_DIR / f"input/data/{item}/test"
                for item in CONFIG.PSP_DATASETS
            ],
            ["lung", "localizer", "bones", "drr", "soft"],
        )

        self.train_dataloader = repeat(
            DataLoader(
                self.train_dataset,
                batch_size=CONFIG.PSP_BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                prefetch_factor=CONFIG.PSP_BATCH_SIZE,
                drop_last=True,
            )
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=CONFIG.PSP_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            prefetch_factor=CONFIG.PSP_BATCH_SIZE,
            drop_last=False,
        )

        # Initialize logger
        self.logger = SummaryWriter(log_dir=str(CONFIG.OUTPUT_DIR / "logs"))

        # Initialize checkpoint dir
        self.checkpoint_dir = CONFIG.OUTPUT_DIR / "checkpoint"

        self.start_iter = 0
        if self.net.resumed:
            self.start_iter = int(os.path.splitext(CONFIG.PSP_CKPT.stem)[0])
            self.optimizer.load_state_dict(ckpt["optim"])

        self.test_dir = CONFIG.OUTPUT_DIR / "test"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def mix(self, name: Literal["train", "test"]):
        self.net.eval()
        output_folder = CONFIG.OUTPUT_DIR / f"mixed_alternate_single/{name}"
        output_folder.mkdir(parents=True, exist_ok=True)
        dataset = self.train_dataset if name == "train" else self.test_dataset
        dataloader = repeat(
            DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=2,
                prefetch_factor=2,
                drop_last=True,
            )
        )

        print(f"Mixing {name} images")
        if len(dataset) < 200:
            n = 500
        else:
            n = 2000
        with torch.no_grad():
            for idx in tqdm(list(range(n))):
                items = next(dataloader)
                imgs = to_device(items)
                output_imgs, w_plus = self.mix_pass(imgs)
                # save_image(
                #     torch.concat(
                #         [
                #             output_imgs["lung1"],
                #             output_imgs["lung2"],
                #             output_imgs["out"],
                #         ],
                #         dim=3,
                #     ),
                #     str(output_folder / f"{str(idx).zfill(6)}.png"),
                #     nrow=1,
                #     normalize=True,
                #     value_range=(-1, 1),
                # )
                save_image(
                    output_imgs["out"],
                    str(output_folder / f"{str(idx).zfill(6)}.png"),
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

    def mix_pass(self, imgs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:

        # Concat image along channel direction
        img_in = torch.cat([imgs[key] for key in CONFIG.PSP_IN], dim=1)
        img_in1 = img_in[0].unsqueeze(0)
        img_in2 = img_in[1].unsqueeze(0)

        img_out: Tensor
        w_plus: Tensor

        img_out, w_plus = self.net.mix((img_in1, img_in2))
        return {"lung1": img_in1, "lung2": img_in2, "out": img_out}, w_plus


def main():
    coach = Coach()
    coach.mix("train")
    coach.mix("test")


if __name__ == "__main__":
    main()

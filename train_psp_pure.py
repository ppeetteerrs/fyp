from utils.config import CONFIG, guard

guard()

# Use StyleGAN output directly

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
from torchvision.utils import save_image
from tqdm import tqdm

from psp.loss_fn.discriminator_loss import DiscriminatorLoss
from psp.loss_fn.id_loss import DiscriminatorIDLoss, IDLoss
from psp.loss_fn.reg_loss import RegLoss
from psp.pSp_pure import pSp
from psp.ranger import Ranger
from utils.dataset import MulticlassImageDataset
from utils.utils import repeat


class Coach:
    def __init__(self):
        ckpt = torch.load(str(CONFIG.PSP_CKPT))
        self.net = pSp(ckpt).to("cuda")

        # Utility functions
        self.id_loss = IDLoss().to("cuda").eval()
        self.lpips_loss = lpips.LPIPS(net="vgg").to("cuda").eval()
        self.reg_loss = RegLoss().to("cuda").eval()
        self.discriminator_id_loss = DiscriminatorIDLoss().to("cuda").eval()
        self.discriminator_loss = DiscriminatorLoss().to("cuda").eval()
        self.ssim = SSIM(11, "mean", 1)

        # Initialize optimizer
        self.optimizer: Optimizer
        if CONFIG.PSP_OPTIM == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.encoder.parameters(),
                lr=CONFIG.PSP_LR,
            )
        else:
            self.optimizer = Ranger(
                self.net.encoder.parameters(),
                lr=CONFIG.PSP_LR,
            )

        # Initialize dataset
        self.train_dataset = MulticlassImageDataset(
            [
                CONFIG.PROJECT_DIR / "input/data" / "lidc/train",
                CONFIG.PROJECT_DIR / "input/data" / "covid_ct/train",
            ],
            ["lung", "localizer", "bones", "drr"],
        )
        self.test_dataset = MulticlassImageDataset(
            [
                CONFIG.PROJECT_DIR / "input/data" / "lidc/test",
                CONFIG.PROJECT_DIR / "input/data" / "covid_ct/test",
            ],
            ["lung", "localizer", "bones", "drr"],
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
            drop_last=True,
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

    def train(self):
        self.net.encoder.train()
        self.net.decoder.eval()

        for step in tqdm(
            range(self.start_iter, CONFIG.PSP_ITER),
            initial=self.start_iter,
            total=CONFIG.PSP_ITER,
            dynamic_ncols=True,
            smoothing=0.01,
        ):
            batch = next(self.train_dataloader)
            lung: Tensor = batch["lung"].to("cuda")
            drr: Tensor = batch["drr"].to("cuda")

            # Configure input and target images
            img_in, img_style, w_plus = self.forward_pass(lung)

            # Calculate loss
            loss, loss_dict = self.calc_loss(img_in, img_style, w_plus)

            # Back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.log_metrics(step, loss_dict, prefix="train")

            # Logging
            if step % CONFIG.PSP_SAMPLE_INTERVAL == 0:
                save_image(
                    torch.concat(
                        [img_in, img_style, drr],
                        dim=3,
                    ),
                    str(CONFIG.OUTPUT_DIR / "sample" / f"{str(step).zfill(6)}.png"),
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

            # Validation related
            if step % CONFIG.PSP_TEST_INTERVAL == 0:
                _ = self.test(step)

            if step % CONFIG.PSP_CKPT_INTERVAL == 00:
                torch.save(
                    {
                        "encoder": self.net.encoder.state_dict(),
                        "decoder": self.net.decoder.state_dict(),
                        "discriminator": self.net.discriminator.state_dict()
                        if self.net.discriminator is not None
                        else None,
                        "optim": self.optimizer.state_dict(),
                        "config": CONFIG,
                    },
                    CONFIG.OUTPUT_DIR / f"checkpoint/{str(step).zfill(6)}.pt",
                )

    def test(self, step: int):
        self.net.eval()

        loss_dicts: List[Dict[str, float]] = []
        test_imgs = []

        for idx, batch in enumerate(self.test_dataloader):
            lung: Tensor = batch["lung"].to("cuda")
            drr: Tensor = batch["drr"].to("cuda")
            with torch.no_grad():

                img_in, img_style, w_plus = self.forward_pass(lung)

                _, loss_dict = self.calc_loss(img_in, img_style, w_plus)
                if idx < CONFIG.PSP_TEST_BATCHES:
                    test_imgs.append(
                        torch.concat(
                            [img_in, img_style, drr],
                            dim=3,
                        )
                    )
            loss_dicts.append(loss_dict)

        save_image(
            torch.concat(test_imgs, dim=0),
            str(self.test_dir / f"{str(step).zfill(6)}.png"),
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        loss_dict = self.agg_loss(loss_dicts)
        self.log_metrics(step, loss_dict, prefix="test")
        return loss_dict

    def forward_pass(self, img_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Concat image along channel direction

        img_style: Tensor
        w_plus: Tensor
        img_style, w_plus = self.net(img_in)
        return img_in, img_style, w_plus

    def calc_loss(
        self,
        img_in: Tensor,
        img_style: Tensor,
        w_plus: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss_dict = {}
        loss: Any = 0.0
        # img_in: (N, in_channel, H, W)
        # img_in_list: list of length in_channel, each (N, 1, H, W)
        # img_in_save: (N, 1, H, W * in_channel)
        # img_out: (N, 1, H, W)
        # img_out_rgb: (N, 3, H, W)
        # img_style: (N, 1, H, W)

        # style_bone = ((img_style + 1) * (bone + 1)) / 2 - 1

        img_style_rgb = img_style.expand(-1, 3, -1, -1)

        # L2 loss
        if CONFIG.PSP_LOSS_L2 > 0:
            loss_l2_style_in = F.mse_loss(img_style, img_in)

            loss_dict["loss_l2_style_in"] = float(loss_l2_style_in)
            loss += loss_l2_style_in * CONFIG.PSP_LOSS_L2

        # Reg loss
        if CONFIG.PSP_LOSS_REG > 0:
            loss_reg = self.reg_loss(w_plus, self.net.latent_avg)
            loss_dict["loss_reg"] = float(loss_reg)
            loss += loss_reg * CONFIG.PSP_LOSS_REG

        # LPIPS loss
        if CONFIG.PSP_LOSS_LPIPS > 0:
            loss_lpips = 0.0
            loss_lpips_item = self.lpips_loss(
                img_style_rgb, img_in.expand(-1, 3, -1, -1)
            )
            loss_lpips = torch.sum(loss_lpips_item) / img_in.shape[0]
            loss_dict["loss_lpips"] = float(loss_lpips)
            loss += loss_lpips * CONFIG.PSP_LOSS_LPIPS

        # Discriminator losses
        if self.net.discriminator is not None:
            d_score: Tensor
            d_features_out: Tensor
            d_score, d_features_out = self.net.discriminator.forward(
                img_style, return_features=True
            )
            # Discriminator ID loss
            if CONFIG.PSP_LOSS_ID_DISCRIMINATOR > 0:
                loss_discriminator_id = 0.0
                _, d_features_loss = self.net.discriminator(
                    img_in, return_features=True
                )
                loss_discriminator_id = self.discriminator_id_loss(
                    d_features_out, d_features_loss
                )
                loss_dict["loss_discriminator_id"] = float(loss_discriminator_id)
                loss += loss_discriminator_id * CONFIG.PSP_LOSS_ID_DISCRIMINATOR

            # Discriminator loss
            if CONFIG.PSP_LOSS_DISCRIMINATOR > 0:
                loss_discriminator: Tensor = self.discriminator_loss(d_score)
                loss_dict["loss_discriminator"] = float(loss_discriminator)
                loss += loss_discriminator * CONFIG.PSP_LOSS_DISCRIMINATOR

        if CONFIG.PSP_LOSS_SSIM > 0:
            loss_ssim = self.ssim(img_style, img_in)
            loss_dict["loss_ssim"] = float(loss_ssim)
            loss += loss_ssim * CONFIG.PSP_LOSS_SSIM

        # Accumulate loss
        loss_dict["loss"] = float(loss)

        return loss, loss_dict

    def log_metrics(self, step: int, loss_dict: Dict[str, float], prefix: str):
        for key, value in loss_dict.items():
            self.logger.add_scalar(f"{prefix}/{key}", value, step)

    @staticmethod
    def agg_loss(dicts: List[Dict[str, float]]) -> Dict[str, float]:
        all_dict: Dict[str, List[float]] = {}
        for dict_item in dicts:
            for key in dict_item:
                all_dict[key] = all_dict.setdefault(key, []) + [dict_item[key]]

        mean_dict: Dict[str, float] = {}
        for key in all_dict:
            if len(all_dict[key]) > 0:
                mean_dict[key] = sum(all_dict[key]) / len(all_dict[key])
        return mean_dict


def main():
    coach = Coach()
    coach.train()


if __name__ == "__main__":
    main()
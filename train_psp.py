import os
from typing import Any, Dict, List, Tuple

import lpips
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from psp.loss_fn.discriminator_loss import DiscriminatorLoss
from psp.loss_fn.id_loss import DiscriminatorIDLoss, IDLoss
from psp.loss_fn.reg_loss import RegLoss
from psp.pSp import pSp
from psp.ranger import Ranger
from utils.config import CONFIG
from utils.img import transform
from utils.lmdb import LMDBImageDataset, covid_ct_indexer_lung
from utils.utils import repeat


class Coach:
    def __init__(self):
        self.net = pSp().to("cuda")

        # Utility functions
        self.resize = torch.nn.AdaptiveAvgPool2d((256, 256)).to("cuda")
        self.id_loss = IDLoss().to("cuda").eval()
        self.lpips_loss = lpips.LPIPS().to("cuda").eval()
        self.reg_loss = RegLoss().to("cuda").eval()
        self.discriminator_id_loss = DiscriminatorIDLoss().to("cuda").eval()
        self.discriminator_loss = DiscriminatorLoss().to("cuda").eval()

        # Initialize optimizer
        self.optimizer: Optimizer
        if CONFIG.PSP_OPTIM == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.encoder.parameters(), lr=CONFIG.PSP_LR
            )
        else:
            self.optimizer = Ranger(self.net.encoder.parameters(), lr=CONFIG.PSP_LR)

        # Initialize dataset
        self.train_dataset = LMDBImageDataset(
            CONFIG.COVID_19_TRAIN_LMDB,
            covid_ct_indexer_lung,
            transform,
        )
        self.test_dataset = LMDBImageDataset(
            CONFIG.COVID_19_TEST_LMDB,
            covid_ct_indexer_lung,
            transform,
            CONFIG.PSP_BATCH_SIZE,
        )
        self.train_dataloader = repeat(
            DataLoader(
                self.train_dataset,
                batch_size=CONFIG.PSP_BATCH_SIZE,
                shuffle=True,
                num_workers=2,
                prefetch_factor=2,
                drop_last=True,
            )
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=CONFIG.PSP_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            drop_last=True,
        )

        # Initialize logger
        self.logger = SummaryWriter(log_dir=str(CONFIG.PSP_OUTPUT_DIR / "logs"))

        # Initialize checkpoint dir
        self.checkpoint_dir = CONFIG.PSP_OUTPUT_DIR / "checkpoint"

        self.start_iter = 0
        if self.net.resumed:
            self.start_iter = int(os.path.splitext(CONFIG.PSP_CKPT.stem)[0])

    def train(self):
        self.net.train()

        for step in tqdm(
            range(self.start_iter, CONFIG.PSP_ITER),
            initial=self.start_iter,
            total=CONFIG.PSP_ITER,
            dynamic_ncols=True,
            smoothing=0.01,
        ):

            # Configure input and target images
            img_in, img_targ, img_out, w_plus = self.forward_pass(
                next(self.train_dataloader)
            )

            # Calculate loss
            loss, loss_dict = self.calc_loss(img_in, img_targ, img_out, w_plus)

            # Back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.log_metrics(step, loss_dict, prefix="train")

            # Logging
            if step % CONFIG.PSP_SAMPLE_INTERVAL == 0:
                save_image(
                    torch.concat([img_in, img_targ, img_out], dim=2),
                    str(CONFIG.PSP_OUTPUT_DIR / "sample" / f"{str(step).zfill(6)}.png"),
                    nrow=CONFIG.PSP_BATCH_SIZE,
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
                    CONFIG.PSP_OUTPUT_DIR / f"checkpoint/{str(step).zfill(6)}.pt",
                )

    def test(self, step: int):
        self.net.eval()

        loss_dicts: List[Dict[str, float]] = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                img_in, img_targ, img_out, w_plus = self.forward_pass(batch)
                _, loss_dict = self.calc_loss(img_in, img_targ, img_out, w_plus)
            loss_dicts.append(loss_dict)

            out_dir = CONFIG.PSP_OUTPUT_DIR / f"test/{step}"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_image(
                torch.concat([img_in, img_targ, img_out], dim=2),
                str(out_dir / f"{str(batch_idx).zfill(6)}.png"),
                nrow=CONFIG.PSP_BATCH_SIZE,
                normalize=True,
                value_range=(-1, 1),
            )

        loss_dict = self.agg_loss(loss_dicts)
        self.log_metrics(step, loss_dict, prefix="test")

        self.net.train()
        return loss_dict

    def forward_pass(
        self, tensors: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Configure input and target images
        img_in, img_targ = tensors
        img_in = img_in.to("cuda")
        img_targ = img_targ.to("cuda")
        img_out, w_plus = self.net.forward(img_in)
        return img_in, img_targ, img_out, w_plus

    def calc_loss(
        self, img_in: Tensor, img_targ: Tensor, img_out: Tensor, w_plus: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss_dict = {}
        loss: Any = 0.0
        # img_in: CT projection, img_targ: localizer, img_out: generated CXR
        img_loss = img_targ if CONFIG.PSP_USE_LOCALIZER else img_in
        img_out_rgb = img_out.expand(-1, 3, -1, -1)
        img_loss_rgb = img_loss.expand(-1, 3, -1, -1)

        # L2 loss
        if CONFIG.PSP_LOSS_L2 > 0:
            # Calculate L2 loss w.r.t.
            loss_l2_in = F.mse_loss(img_out, img_in)
            loss_l2_targ = (
                F.mse_loss(img_out, img_targ) if CONFIG.PSP_USE_LOCALIZER else 0
            )
            targ_weight = CONFIG.PSP_LOCALIZER_WEIGHT if CONFIG.PSP_USE_LOCALIZER else 0
            loss_l2 = loss_l2_in * (1 - targ_weight) + loss_l2_targ * targ_weight

            loss_dict["loss_l2"] = float(loss_l2)
            loss += loss_l2 * CONFIG.PSP_LOSS_L2

        # ID loss
        if CONFIG.PSP_LOSS_ID > 0:
            loss_id = self.id_loss.forward(img_out_rgb, img_loss_rgb)
            loss_dict["loss_id"] = float(loss_id)
            loss += loss_id * CONFIG.PSP_LOSS_ID

        # Reg loss
        if CONFIG.PSP_LOSS_REG > 0:
            loss_reg = self.reg_loss.forward(w_plus, self.net.latent_avg)
            loss_dict["loss_reg"] = float(loss_reg)
            loss += loss_reg * CONFIG.PSP_LOSS_REG

        # # Reg loss
        if CONFIG.PSP_LOSS_LPIPS > 0:
            loss_lpips: Any = self.lpips_loss.forward(img_out_rgb, img_loss_rgb)
            loss_lpips = torch.sum(loss_lpips) / img_out_rgb.shape[0]
            loss_dict["loss_lpips"] = float(loss_lpips)
            loss += loss_lpips * CONFIG.PSP_LOSS_LPIPS

        # Discriminator losses
        if self.net.discriminator is not None:
            d_score: Tensor
            d_features_out: Tensor
            d_score, d_features_out = self.net.discriminator.forward(
                img_out, return_features=True
            )
            # Discriminator ID loss
            if CONFIG.PSP_LOSS_ID_DISCRIMINATOR > 0:
                _, d_features_loss = self.net.discriminator.forward(
                    img_loss, return_features=True
                )
                loss_discriminator_id = self.discriminator_id_loss.forward(
                    d_features_out, d_features_loss
                )
                loss_dict["loss_discriminator_id"] = float(loss_discriminator_id)
                loss += loss_discriminator_id * CONFIG.PSP_LOSS_ID_DISCRIMINATOR

            # Discriminator loss
            if CONFIG.PSP_LOSS_DISCRIMINATOR > 0:
                loss_discriminator: Tensor = self.discriminator_loss.forward(d_score)
                loss_dict["loss_discriminator"] = float(loss_discriminator)
                loss += loss_discriminator * CONFIG.PSP_LOSS_DISCRIMINATOR

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

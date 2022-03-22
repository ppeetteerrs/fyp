from utils.config import CONFIG, guard

guard()


import os
from typing import Any, Dict, List, Literal, Tuple

import lpips
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeometry.losses import SSIM
from torchvision.utils import save_image
from tqdm import tqdm

from psp.loss_fn.discriminator_loss import (DenseDiscriminator,
                                            DiscriminatorLoss, IDLoss)
from psp.loss_fn.reg_loss import RegLoss
from psp.pSp_deep import pSp
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
            length=CONFIG.PSP_BATCH_SIZE * CONFIG.PSP_TEST_BATCHES,
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
            batch = to_device(next(self.train_dataloader))

            # Configure input and target images
            imgs, w_plus = self.forward_pass(batch)

            # Calculate loss
            loss, loss_dict = self.calc_loss(imgs, w_plus)

            # Back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.log_metrics(step, loss_dict, prefix="train")

            # Logging
            if step % CONFIG.PSP_SAMPLE_INTERVAL == 0:
                save_image(
                    torch.concat(
                        [
                            imgs["lung"],
                            imgs["out"],
                            imgs["drr"],
                            imgs["bones"],
                            imgs["soft"],
                            imgs["combined"],
                        ],
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

            if step % CONFIG.PSP_CKPT_INTERVAL == 0:
                torch.save(
                    {
                        "latent_avg": self.net.latent_avg,
                        "encoder": self.net.encoder.state_dict(),
                        "decoder": self.net.decoder.state_dict(),
                        "optim": self.optimizer.state_dict(),
                        "config": CONFIG,
                    },
                    CONFIG.OUTPUT_DIR / f"checkpoint/{str(step).zfill(6)}.pt",
                )

    def test(self, step: int):
        self.net.eval()

        loss_dicts: List[Dict[str, float]] = []
        test_imgs = []

        for _, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                batch = to_device(batch)

                imgs, w_plus = self.forward_pass(batch)

                _, loss_dict = self.calc_loss(imgs, w_plus)
                test_imgs.append(
                    torch.concat(
                        [
                            imgs["lung"],
                            imgs["out"],
                            imgs["drr"],
                            imgs["bones"],
                            imgs["soft"],
                            imgs["combined"],
                        ],
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

    def generate(self, name: Literal["train", "test"]):
        self.net.eval()
        output_folder = CONFIG.OUTPUT_DIR / f"generated/{name}"
        output_folder.mkdir(parents=True, exist_ok=True)
        dataset = self.train_dataset if name == "train" else self.test_dataset
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            prefetch_factor=1,
            drop_last=False,
        )

        print(f"Generating {name} images")
        with torch.no_grad():
            for idx, item in tqdm(enumerate(dataloader)):
                imgs = to_device(item)
                output_imgs, w_plus = self.forward_pass(imgs)
                save_image(
                    output_imgs["out"],
                    str(output_folder / f"{str(idx).zfill(6)}.png"),
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

    def mix(self, name: Literal["train", "test"]):
        self.net.eval()
        output_folder = CONFIG.OUTPUT_DIR / f"mixed/{name}"
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
        with torch.no_grad():
            for idx in tqdm(list(range(len(dataset) * 3))):
                items = next(dataloader)
                imgs = to_device(items)
                output_imgs, w_plus = self.mix_pass(imgs)
                save_image(
                    torch.concat(
                        [
                            output_imgs["lung1"],
                            output_imgs["lung2"],
                            output_imgs["out"],
                        ],
                        dim=3,
                    ),
                    str(output_folder / f"{str(idx).zfill(6)}.png"),
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

    def forward_pass(self, imgs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:

        # Concat image along channel direction
        img_in = torch.cat([imgs[key] for key in CONFIG.PSP_IN], dim=1)

        img_out: Tensor
        w_plus: Tensor

        img_out, w_plus = self.net(img_in)
        return {**imgs, "out": img_out}, w_plus

    def mix_pass(self, imgs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:

        # Concat image along channel direction
        img_in = torch.cat([imgs[key] for key in CONFIG.PSP_IN], dim=1)
        img_in1 = img_in[0].unsqueeze(0)
        img_in2 = img_in[1].unsqueeze(0)

        img_out: Tensor
        w_plus: Tensor

        img_out, w_plus = self.net.mix((img_in1, img_in2))
        return {"lung1": img_in1, "lung2": img_in2, "out": img_out}, w_plus

    def calc_loss(
        self,
        imgs: Dict[str, Tensor],
        w_plus: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss_dict: Dict[str, float] = {}
        loss: Any = 0.0

        for img1, img2, weight in CONFIG.PSP_LOSS_L2:
            loss_l2 = F.mse_loss(imgs[img1], imgs[img2])
            loss_dict[f"loss_l2_{img1}:{img2}"] = float(loss_l2)
            loss += loss_l2 * weight

        if CONFIG.PSP_LOSS_DISCRIMINATOR > 0:
            score, _ = self.discriminator(imgs["out"])
            loss_discriminator, prob = self.discriminator_loss(score)
            loss_dict["loss_discriminator"] = float(loss_discriminator)
            loss_dict["prob_discriminator"] = float(prob)
            loss += loss_discriminator * CONFIG.PSP_LOSS_DISCRIMINATOR

        for img1, img2, weight in CONFIG.PSP_LOSS_ID:
            _, features1 = self.discriminator(imgs[img1])
            _, features2 = self.discriminator(imgs[img2])
            loss_id = self.id_loss.forward(features1, features2)
            loss_dict[f"loss_id_{img1}:{img2}"] = float(loss_id)
            loss += loss_id * weight

        # Reg loss
        if CONFIG.PSP_LOSS_REG > 0:
            loss_reg = self.reg_loss(w_plus, self.net.latent_avg)
            loss_dict["loss_reg"] = float(loss_reg)
            loss += loss_reg * CONFIG.PSP_LOSS_REG

        for img1, img2, weight in CONFIG.PSP_LOSS_LPIPS:
            loss_lpips = self.lpips_loss(
                imgs[img1].expand(-1, 3, -1, -1), imgs[img2].expand(-1, 3, -1, -1)
            )
            loss_lpips = loss_lpips.mean()
            loss_dict[f"loss_lpips_{img1}:{img2}"] = float(loss_lpips)
            loss += loss_lpips * weight

        for img1, img2, weight in CONFIG.PSP_LOSS_SSIM:
            loss_ssim = self.ssim(imgs[img1], imgs[img2])
            loss_ssim = loss_ssim.mean()
            loss_dict[f"loss_ssim_{img1}:{img2}"] = float(loss_ssim)
            loss += loss_ssim * weight

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
    coach.generate("train")
    coach.generate("test")
    coach.mix("train")
    coach.mix("test")


if __name__ == "__main__":
    main()

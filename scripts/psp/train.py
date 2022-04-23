"""Train pSp"""

import os
from pathlib import Path
from typing import Dict, Tuple, Union, cast

import lpips
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import repeat, to_device
from utils.cli import OPTIONS, save_options
from utils.cli.psp import PSPArch, PSPTrain
from utils.dataset import LMDBImageDataset

from psp import pSp
from psp.loss import IDLoss, RegLoss
from psp.ranger import Ranger

ARCH_OPTIONS = cast(PSPArch, OPTIONS.arch)
TRAIN_OPTIONS = cast(PSPTrain, ARCH_OPTIONS.cmd)


class Task:
    def __init__(self):
        """Class that wraps a pSp model to perform training, generation or mixing."""

        # Load checkpoint
        ckpt = torch.load(ARCH_OPTIONS.ckpt)
        self.net = pSp(
            ckpt,
            use_mean=ARCH_OPTIONS.use_mean,
            e_in_channel=len(ARCH_OPTIONS.inputs),
            e_resolution=ARCH_OPTIONS.input_resolution,
            g_resolution=ARCH_OPTIONS.output_resolution,
            g_latent_dim=ARCH_OPTIONS.latent_dim,
            g_n_mlp=ARCH_OPTIONS.n_mlp,
            g_lr_mlp_mult=ARCH_OPTIONS.lr_mlp_mult,
            g_channels=ARCH_OPTIONS.channels_map,
            g_blur_kernel=ARCH_OPTIONS.blur_kernel,
        ).to("cuda")

        # Loss functions
        self.lpips_loss = lpips.LPIPS(net="vgg").to("cuda").eval()
        self.reg_loss = RegLoss(self.net.latent_avg).to("cuda").eval()
        self.id_loss = IDLoss(TRAIN_OPTIONS.id_ckpt).to("cuda").eval()

        # Optimizer
        self.optimizer = Ranger(
            self.net.encoder.parameters(),
            lr=TRAIN_OPTIONS.lr,
        )

        # Initialize dataset
        self.dataset = LMDBImageDataset(
            TRAIN_OPTIONS.dataset,
            ARCH_OPTIONS.classes,
        )
        self.samples = LMDBImageDataset(
            TRAIN_OPTIONS.dataset,
            ARCH_OPTIONS.classes,
            length=TRAIN_OPTIONS.sample_size,
        )

        self.dataloader = repeat(
            DataLoader(
                self.dataset,
                batch_size=TRAIN_OPTIONS.batch_size,
                shuffle=True,
                num_workers=2,
                prefetch_factor=TRAIN_OPTIONS.batch_size,
                drop_last=True,
            )
        )
        self.sample_loader = repeat(
            DataLoader(
                self.samples,
                batch_size=TRAIN_OPTIONS.sample_size,
                shuffle=False,
                num_workers=2,
                prefetch_factor=TRAIN_OPTIONS.sample_size,
                drop_last=True,
            )
        )

        # If using pretrained pSp
        if self.net.resumed:
            if "step" in ckpt:
                self.start_iter = ckpt["iter"]
            else:
                ckpt_name = os.path.basename(ARCH_OPTIONS.ckpt)
                self.start_iter = int(os.path.splitext(ckpt_name)[0])

            self.optimizer.load_state_dict(ckpt["optim"])
        else:
            self.start_iter = 0

        # Start wandb
        wandb.init(
            project="FYP",
            entity="ppeetteerrs",
            group="psp",
            job_type="train",
            name=OPTIONS.name,
            config=OPTIONS.to_dict(),
        )

    def forward(self, imgs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Forward propagate input images and add output image to dictionary.
        Also returns $W+$ style vectors.
        """

        # Concat image along channel direction
        img_in = torch.cat([imgs[key] for key in ARCH_OPTIONS.inputs], dim=1)

        img_out, w_plus = self.net(img_in, "generate")
        return {**imgs, "out": img_out}, w_plus

    def loss(
        self,
        imgs: Dict[str, Tensor],
        w_plus: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Accumulates losses from each loss component according to loss specs
        from CLI options.
        """

        loss_dict: Dict[str, float] = {}
        loss: Union[float, Tensor] = 0.0

        for img, truth, weight in TRAIN_OPTIONS.l2_spec:
            loss_l2 = F.mse_loss(imgs[img], imgs[truth])
            loss_dict[f"l2_{img}:{truth}"] = float(loss_l2)
            loss += loss_l2 * weight

        for img, truth, weight in TRAIN_OPTIONS.id_spec:
            loss_id = self.id_loss(
                imgs[img].expand(-1, 3, -1, -1), imgs[truth].expand(-1, 3, -1, -1)
            )
            loss_dict[f"id_{img}:{truth}"] = float(loss_id)
            loss += loss_id * weight

        # Reg loss
        if TRAIN_OPTIONS.reg > 0:
            loss_reg = self.reg_loss(w_plus)
            loss_dict["reg"] = float(loss_reg)
            loss += loss_reg * TRAIN_OPTIONS.reg

        for img, truth, weight in TRAIN_OPTIONS.lpips_spec:
            loss_lpips = self.lpips_loss(
                imgs[img].expand(-1, 3, -1, -1), imgs[truth].expand(-1, 3, -1, -1)
            ).mean()
            loss_dict[f"lpips_{img}:{truth}"] = float(loss_lpips)
            loss += loss_lpips * weight

        # Accumulate loss
        loss_dict["loss"] = float(loss)

        return cast(Tensor, loss), loss_dict

    def train(self):
        """Trains pSp model"""

        # Only trains encoder
        self.net.encoder.train()
        self.net.decoder.eval()

        # Training loop
        for step in tqdm(
            range(self.start_iter, TRAIN_OPTIONS.iterations + 1),
            initial=self.start_iter,
            total=TRAIN_OPTIONS.iterations + 1,
            dynamic_ncols=True,
            smoothing=0.01,
        ):
            batch = to_device(next(self.dataloader))

            # Configure input and target images
            imgs, w_plus = self.forward(batch)

            # Calculate loss
            loss, loss_dict = self.loss(imgs, w_plus)

            # Back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            wandb.log(loss_dict, step=step)

            if step % TRAIN_OPTIONS.sample_interval == 0:
                sample_imgs, _ = self.forward(to_device(next(self.sample_loader)))
                key = "_".join(sample_imgs.keys())
                img = torch.cat(list(sample_imgs.values()), dim=3)

                wandb.log(
                    {
                        key: wandb.Image(
                            make_grid(
                                img,
                                nrow=1,
                                normalize=True,
                                value_range=(-1, 1),
                            )
                        )
                    },
                    step=step,
                )

            # Validation related
            if step % TRAIN_OPTIONS.ckpt_interval == 0:
                torch.save(
                    {
                        "latent_avg": self.net.latent_avg,
                        "encoder": self.net.encoder.state_dict(),
                        "decoder": self.net.decoder.state_dict(),
                        "optim": self.optimizer.state_dict(),
                        "iter": step,
                    },
                    OPTIONS.output_dir / f"{str(step).zfill(6)}.pt",
                )


def psp_train():
    save_options()
    Task().train()

"""Train StyleGAN"""
from typing import cast

import torch
import wandb
from stylegan2_torch import Discriminator, Generator
from stylegan2_torch.loss import d_loss as get_d_loss
from stylegan2_torch.loss import d_reg_loss as get_d_reg_loss
from stylegan2_torch.loss import g_loss as get_g_loss
from stylegan2_torch.loss import g_reg_loss as get_g_reg_loss
from stylegan2_torch.utils import mixing_noise
from torch import distributed, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import accumulate, repeat
from utils.cli import OPTIONS, save_options
from utils.cli.stylegan import StyleGANArch, StyleGANTrain
from utils.dataset import LMDBImageDataset
from utils.distributed import reduce_loss_dict, reduce_sum, setup_distributed

ARCH_OPTIONS = cast(StyleGANArch, OPTIONS.arch)
TRAIN_OPTIONS = cast(StyleGANTrain, ARCH_OPTIONS.cmd)


class Task:
    def __init__(self) -> None:

        # Use deterministic algorithms
        benchmark = True

        # Setup distributed process
        self.local_rank, self.world_size = setup_distributed()

        # Create models
        generator = Generator(ARCH_OPTIONS.output_resolution).to("cuda")
        discriminator = Discriminator(ARCH_OPTIONS.output_resolution).to("cuda")
        self.g_ema = Generator(ARCH_OPTIONS.output_resolution).to("cuda")
        self.g_ema.eval()

        # Initialize ema model
        accumulate(self.g_ema, generator, 0)

        # Setup optimizers
        g_reg_ratio = TRAIN_OPTIONS.g_reg_interval / (TRAIN_OPTIONS.g_reg_interval + 1)
        d_reg_ratio = TRAIN_OPTIONS.d_reg_interval / (TRAIN_OPTIONS.d_reg_interval + 1)

        self.g_optim = optim.Adam(
            generator.parameters(),
            lr=TRAIN_OPTIONS.lr * g_reg_ratio,
            betas=(0, 0.99**g_reg_ratio),
        )
        self.d_optim = optim.Adam(
            discriminator.parameters(),
            lr=TRAIN_OPTIONS.lr * d_reg_ratio,
            betas=(0, 0.99**d_reg_ratio),
        )

        # Load checkpoint
        self.sample_z = None
        self.start_iter = 0
        if ARCH_OPTIONS.ckpt is not None:
            ckpt = torch.load(ARCH_OPTIONS.ckpt)

            generator.load_state_dict(ckpt["g"])
            discriminator.load_state_dict(ckpt["d"])
            self.g_ema.load_state_dict(ckpt["g_ema"])

            self.g_optim.load_state_dict(ckpt["g_optim"])
            self.d_optim.load_state_dict(ckpt["d_optim"])

            self.start_iter = ckpt["iter"]
            self.sample_z = ckpt["sample_z"].to("cuda")
        else:
            self.start_iter = 0
            self.sample_z = torch.randn(
                TRAIN_OPTIONS.sample_size, ARCH_OPTIONS.latent_dim, device="cuda"
            )

        # Setup distributed models
        self.generator = DistributedDataParallel(
            generator,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
        )

        self.discriminator = DistributedDataParallel(
            discriminator,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
        )
        self.g_module = cast(Generator, self.generator.module)
        self.d_module = cast(Discriminator, self.discriminator.module)

        # Setup Dataloader
        self.dataset = LMDBImageDataset(TRAIN_OPTIONS.dataset)
        self.loader = repeat(
            DataLoader(
                self.dataset,
                batch_size=TRAIN_OPTIONS.batch_size,
                sampler=DistributedSampler(self.dataset, shuffle=True),
                drop_last=True,
                num_workers=2,
                prefetch_factor=TRAIN_OPTIONS.batch_size,
            )
        )

    def train(self):

        # Initialize logging tensors
        d_reg_loss = torch.tensor(0.0, device="cuda")
        g_reg_loss = torch.tensor(0.0, device="cuda")
        path_lengths = torch.tensor(0.0, device="cuda")
        mean_path_length = 0
        mean_path_length_avg = 0
        loss_dict = {}

        # Setup processes
        if self.local_rank == 0:
            pbar = tqdm(
                range(self.start_iter, TRAIN_OPTIONS.iterations + 1),
                initial=self.start_iter,
                total=TRAIN_OPTIONS.iterations + 1,
                dynamic_ncols=True,
                smoothing=0.01,
            )
            self.writer = SummaryWriter(f"tb_logs/{OPTIONS.name}")
            self.writer.add_text("Options", f"```yaml\n{OPTIONS.dumps_yaml()}\n```")
        else:
            pbar = range(self.start_iter, TRAIN_OPTIONS.iterations + 1)

        for step in pbar:

            # Sample real image
            real_img = next(self.loader).to("cuda")

            # Only trains discriminator
            self.generator.requires_grad_(False)
            self.discriminator.requires_grad_(True)

            # Get noise style(s)
            noise = mixing_noise(
                TRAIN_OPTIONS.batch_size,
                ARCH_OPTIONS.latent_dim,
                TRAIN_OPTIONS.mixing,
                "cuda",
            )
            fake_img = self.generator(noise)

            # Train discriminator
            fake_pred = self.discriminator(fake_img)
            real_pred = self.discriminator(real_img)
            d_loss = get_d_loss(real_pred, fake_pred)

            loss_dict["d_loss"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            # Regularize discriminator
            if step % TRAIN_OPTIONS.d_reg_interval == 0:
                real_img.requires_grad = True

                real_pred = self.discriminator(real_img)
                d_reg_loss = get_d_reg_loss(real_pred, real_img)

                self.discriminator.zero_grad()

                # Trigger gradient reduce
                (
                    TRAIN_OPTIONS.r1 / 2 * d_reg_loss * TRAIN_OPTIONS.d_reg_interval
                    + 0 * real_pred[0]
                ).backward()

                self.d_optim.step()

            loss_dict["d_reg_loss"] = d_reg_loss

            # Only trains generator
            self.generator.requires_grad_(True)
            self.discriminator.requires_grad_(False)

            # Get noise style(s)
            noise = mixing_noise(
                TRAIN_OPTIONS.batch_size,
                ARCH_OPTIONS.latent_dim,
                TRAIN_OPTIONS.mixing,
                "cuda",
            )
            fake_img = self.generator(noise)

            # Train generator
            fake_pred = self.discriminator(fake_img)
            g_loss = get_g_loss(fake_pred)

            loss_dict["g_loss"] = g_loss

            self.generator.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            # Regularize generator
            if step % TRAIN_OPTIONS.g_reg_interval == 0:
                path_batch_size = max(
                    1, TRAIN_OPTIONS.batch_size // TRAIN_OPTIONS.path_batch_shrink
                )
                noise = mixing_noise(
                    path_batch_size,
                    ARCH_OPTIONS.latent_dim,
                    TRAIN_OPTIONS.mixing,
                    "cuda",
                )
                fake_img, latents = self.generator(noise, return_latents=True)

                g_reg_loss, mean_path_length, path_lengths = get_g_reg_loss(
                    fake_img, latents, mean_path_length
                )

                self.generator.zero_grad()
                weighted_path_loss = (
                    TRAIN_OPTIONS.path_reg * TRAIN_OPTIONS.g_reg_interval * g_reg_loss
                    + 0 * fake_img[0, 0, 0, 0]
                )

                weighted_path_loss.backward()

                self.g_optim.step()

                mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / distributed.get_world_size()
                )

            loss_dict["g_reg_loss"] = g_reg_loss
            loss_dict["path_length"] = path_lengths.mean()

            accumulate(self.g_ema, self.g_module)

            loss_reduced = reduce_loss_dict(loss_dict)

            if isinstance(pbar, tqdm):
                self.writer.add_scalar(
                    "d_loss", loss_reduced["d_loss"].mean().item(), step
                )
                self.writer.add_scalar(
                    "g_loss", loss_reduced["g_loss"].mean().item(), step
                )
                self.writer.add_scalar(
                    "d_reg_loss", loss_reduced["d_reg_loss"].mean().item(), step
                )
                self.writer.add_scalar(
                    "g_reg_loss", loss_reduced["g_reg_loss"].mean().item(), step
                )
                self.writer.add_scalar(
                    "real_score", loss_reduced["real_score"].mean().item(), step
                )
                self.writer.add_scalar(
                    "fake_score", loss_reduced["fake_score"].mean().item(), step
                )
                self.writer.add_scalar("mean_path", mean_path_length_avg, step)

                if step % TRAIN_OPTIONS.sample_interval == 0:
                    with torch.no_grad():
                        self.g_ema.eval()
                        sample = self.g_ema([self.sample_z])
                        self.writer.add_image(
                            "sample",
                            make_grid(
                                sample,
                                nrow=int(TRAIN_OPTIONS.sample_size**0.5),
                                normalize=True,
                                value_range=(-1, 1),
                            ),
                            step,
                        )

                if step % TRAIN_OPTIONS.ckpt_interval == 0:
                    torch.save(
                        {
                            "g": self.g_module.state_dict(),
                            "d": self.d_module.state_dict(),
                            "g_ema": self.g_ema.state_dict(),
                            "g_optim": self.g_optim.state_dict(),
                            "d_optim": self.d_optim.state_dict(),
                            "sample_z": self.sample_z,
                            "iter": step,
                        },
                        f"{OPTIONS.output_dir}/{str(step).zfill(6)}.pt",
                    )


def stylegan_train():
    save_options()
    Task().train()

from utils.config import CONFIG, guard

guard()


import os
from typing import Generator as GeneratorType
from typing import Optional, cast

import torch
from stylegan2_torch import Discriminator, Generator
from stylegan2_torch.utils import (
    accumulate,
    d_logistic_loss,
    d_r1_loss,
    g_nonsaturating_loss,
    g_path_regularize,
    mixing_noise,
)
from torch import distributed, optim
from torch.backends import cudnn
from torch.distributed import init_process_group
from torch.functional import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from tqdm import tqdm

from utils.distributed import reduce_loss_dict, reduce_sum
from utils.img import transform
from utils.lmdb import LMDBImageDataset, chexpert_indexer
from utils.utils import repeat


def train(
    loader: GeneratorType[Tensor, None, None],
    generator: DistributedDataParallel,
    discriminator: DistributedDataParallel,
    g_optim: Optimizer,
    d_optim: Optimizer,
    g_ema: Generator,
    sample_z: Optional[Tensor],
    start_iter: int,
):

    if distributed.get_rank() == 0:
        pbar = tqdm(
            range(start_iter, CONFIG.STYLEGAN_ITER),
            initial=start_iter,
            total=CONFIG.STYLEGAN_ITER,
            dynamic_ncols=True,
            smoothing=0.01,
        )
    else:
        pbar = range(start_iter, CONFIG.STYLEGAN_ITER)

    logger = SummaryWriter(log_dir=str(CONFIG.OUTPUT_DIR / "logs"))

    # Initialize tensors
    r1_loss = torch.tensor(0.0, device="cuda")
    path_loss = torch.tensor(0.0, device="cuda")
    path_lengths = torch.tensor(0.0, device="cuda")
    mean_path_length = 0
    mean_path_length_avg = 0
    loss_dict = {}

    g_module = cast(Generator, generator.module)
    d_module = cast(Discriminator, discriminator.module)

    sample_z = (
        sample_z
        if sample_z is not None
        else torch.randn(CONFIG.STYLEGAN_SAMPLES, CONFIG.LATENT_DIM, device="cuda")
    )

    for idx in pbar:

        # Sample real image
        real_img = next(loader)[0].to("cuda")

        # Only trains discriminator
        generator.requires_grad_(False)
        discriminator.requires_grad_(True)

        # Get noise style(s)
        noise = mixing_noise(
            CONFIG.STYLEGAN_BATCH, CONFIG.LATENT_DIM, CONFIG.STYLEGAN_MIXING, "cuda"
        )
        fake_img = generator(noise)

        # Train discriminator
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        # Regularize discriminator
        if idx % CONFIG.STYLEGAN_D_REG_INTERVAL == 0:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (
                CONFIG.STYLEGAN_R1 / 2 * r1_loss * CONFIG.STYLEGAN_D_REG_INTERVAL
                + 0 * real_pred[0]
            ).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # Only trains generator
        generator.requires_grad_(True)
        discriminator.requires_grad_(False)

        # Get noise style(s)
        noise = mixing_noise(
            CONFIG.STYLEGAN_BATCH, CONFIG.LATENT_DIM, CONFIG.STYLEGAN_MIXING, "cuda"
        )
        fake_img = generator(noise)

        # Train generator
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Regularize generator
        if idx % CONFIG.STYLEGAN_G_REG_INTERVAL == 0:
            path_batch_size = max(
                1, CONFIG.STYLEGAN_BATCH // CONFIG.STYLEGAN_PATH_BATCH_SHRINK
            )
            noise = mixing_noise(
                path_batch_size, CONFIG.LATENT_DIM, CONFIG.STYLEGAN_MIXING, "cuda"
            )
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = (
                CONFIG.STYLEGAN_PATH_REG * CONFIG.STYLEGAN_G_REG_INTERVAL * path_loss
            )

            if CONFIG.STYLEGAN_PATH_BATCH_SHRINK:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / distributed.get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module)

        loss_reduced = reduce_loss_dict(loss_dict)

        if isinstance(pbar, tqdm):
            logger.add_scalar(f"train/d", loss_reduced["d"].mean().item(), idx)
            logger.add_scalar(f"train/g", loss_reduced["g"].mean().item(), idx)
            logger.add_scalar(f"train/r1", loss_reduced["r1"].mean().item(), idx)
            logger.add_scalar(f"train/path", loss_reduced["path"].mean().item(), idx)
            logger.add_scalar(f"train/mean_path", mean_path_length_avg, idx)

            if idx % 1000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        f"{CONFIG.OUTPUT_DIR}/sample/{str(idx).zfill(6)}.png",
                        nrow=int(CONFIG.STYLEGAN_SAMPLES**0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

            if idx % 5000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "sample_z": sample_z,
                    },
                    f"{CONFIG.OUTPUT_DIR}/checkpoint/{str(idx).zfill(6)}.pt",
                )


if __name__ == "__main__":
    cudnn.benchmark = True

    # Setup distributed processes
    init_process_group(backend="nccl", init_method="env://")
    assert distributed.is_available(), "torch.distributed not available"
    assert distributed.is_initialized(), "torch.distributed not initialized"
    torch.cuda.set_device(distributed.get_rank())
    distributed.barrier()

    # Create models
    generator = Generator(CONFIG.RESOLUTION).to("cuda")
    discriminator = Discriminator(CONFIG.RESOLUTION).to("cuda")
    g_ema = Generator(CONFIG.RESOLUTION).to("cuda")
    g_ema.eval()

    # Initialize ema model
    accumulate(g_ema, generator, 0)

    # Setup optimizers
    g_reg_ratio = CONFIG.STYLEGAN_G_REG_INTERVAL / (CONFIG.STYLEGAN_G_REG_INTERVAL + 1)
    d_reg_ratio = CONFIG.STYLEGAN_D_REG_INTERVAL / (CONFIG.STYLEGAN_D_REG_INTERVAL + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=CONFIG.STYLEGAN_LR * g_reg_ratio,
        betas=(0, 0.99**g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=CONFIG.STYLEGAN_LR * d_reg_ratio,
        betas=(0, 0.99**d_reg_ratio),
    )

    # Load checkpoint
    sample_z = None
    start_iter = 0
    if CONFIG.STYLEGAN_CKPT is not None:
        ckpt = torch.load(CONFIG.STYLEGAN_CKPT)

        ckpt_name = os.path.basename(CONFIG.STYLEGAN_CKPT)
        start_iter = int(os.path.splitext(ckpt_name)[0])

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        sample_z = ckpt["sample_z"].to("cuda")

    # Setup distributed models
    generator = DistributedDataParallel(
        generator,
        device_ids=[distributed.get_rank()],
        output_device=distributed.get_rank(),
        broadcast_buffers=False,
    )

    discriminator = DistributedDataParallel(
        discriminator,
        device_ids=[distributed.get_rank()],
        output_device=distributed.get_rank(),
        broadcast_buffers=False,
    )

    # Setup Dataloader
    dataset = LMDBImageDataset(CONFIG.CHEXPERT_TRAIN_LMDB, chexpert_indexer, transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.STYLEGAN_BATCH,
        sampler=sampler,
        drop_last=True,
        num_workers=2,
        prefetch_factor=CONFIG.STYLEGAN_BATCH,
    )

    train(
        repeat(loader),
        generator,
        discriminator,
        g_optim,
        d_optim,
        g_ema,
        sample_z,
        start_iter,
    )

import os
from typing import Iterable

import torch
from torch import distributed, optim
from torch.distributed import init_process_group
from torch.functional import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, utils
from tqdm import tqdm

from stylegan.dataset import MultiResolutionDataset
from stylegan.discriminator.discriminator import Discriminator
from stylegan.distributed import reduce_loss_dict, reduce_sum
from stylegan.generator.generator import Generator
from stylegan.parsers import TrainArgs, get_train_args
from stylegan.utils import (
    accumulate,
    d_logistic_loss,
    d_r1_loss,
    g_nonsaturating_loss,
    g_path_regularize,
    mixing_noise,
)


def train(
    args: TrainArgs,
    loader_: DataLoader[int],
    generator: DistributedDataParallel,
    discriminator: DistributedDataParallel,
    g_optim: Optimizer,
    d_optim: Optimizer,
    g_ema: Generator,
):
    loader: Iterable[Tensor] = iter(loader_)

    pbar = range(args.start_iter, args.iter)

    if distributed.get_rank() == 0:
        pbar = tqdm(
            pbar,
            initial=args.start_iter,
            total=args.iter,
            dynamic_ncols=True,
            smoothing=0.05,
        )

    # Initialize tensors
    r1_loss = torch.tensor(0.0, device=args.device)
    path_loss = torch.tensor(0.0, device=args.device)
    path_lengths = torch.tensor(0.0, device=args.device)
    mean_path_length = 0
    mean_path_length_avg = 0
    loss_dict = {}

    g_module = generator.module
    d_module = discriminator.module

    sample_z = torch.randn(args.n_sample, args.latent_dim, device=args.device)

    for idx in pbar:

        # Sample real image
        real_img = next(loader).to(args.device)

        # Only trains discriminator
        generator.requires_grad_(False)
        discriminator.requires_grad_(True)

        # Get noise style(s)
        noise = mixing_noise(args.batch, args.latent_dim, args.mixing, args.device)
        fake_img, _ = generator(noise)

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
        if idx % args.d_reg_every == 0:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # Only trains generator
        generator.requires_grad_(True)
        discriminator.requires_grad_(False)

        # Get noise style(s)
        noise = mixing_noise(args.batch, args.latent_dim, args.mixing, args.device)
        fake_img, _ = generator(noise)

        # Train generator
        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Regularize generator
        if idx % args.g_reg_every == 0:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(
                path_batch_size, args.latent_dim, args.mixing, args.device
            )
            fake_img, latents = generator(noise)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
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

        if distributed.get_rank() == 0:
            pbar: tqdm
            pbar.set_description(
                (
                    f"d: {loss_reduced['d'].mean().item():.4f}; "
                    f"g: {loss_reduced['g'].mean().item():.4f}; "
                    f"r1: {loss_reduced['r1'].mean().item():.4f}; "
                    f"path: {loss_reduced['path'].mean().item():.4f}; "
                    f"mean path: {mean_path_length_avg:.4f}; "
                )
            )

            if idx % 500 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        f"results/sample/{str(idx).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
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
                        "args": args,
                    },
                    f"results/checkpoint/{str(idx).zfill(6)}.pt",
                )


if __name__ == "__main__":
    # Parse arguments
    args = get_train_args()

    # Setup distributed processes
    init_process_group(backend="nccl", init_method="env://")
    assert distributed.is_available(), "torch.distributed not available"
    assert distributed.is_initialized(), "torch.distributed not initialized"
    torch.cuda.set_device(distributed.get_rank())
    distributed.barrier()

    # Create models
    generator = Generator(args).to(args.device)
    discriminator = Discriminator(args).to(args.device)
    print(discriminator)
    g_ema = Generator(args).to(args.device)
    g_ema.eval()

    # Initialize ema model
    accumulate(g_ema, generator, 0)

    # Setup optimizers
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0, 0.99 ** d_reg_ratio),
    )

    # Load checkpoint
    if args.ckpt is not None:
        print("Loading model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, _: storage)

        ckpt_name = os.path.basename(args.ckpt)
        args.start_iter = int(os.path.splitext(ckpt_name)[0])

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * args.n_colors, [0.5] * args.n_colors, inplace=True
            ),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        drop_last=True,
    )

    # default `log_dir` is "runs" - we'll be more specific here

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema)

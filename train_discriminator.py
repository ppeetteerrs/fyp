from utils.config import CONFIG, guard

guard()

# Use StyleGAN output directly

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from stylegan2_torch import Generator
from stylegan2_torch.utils import mixing_noise
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from psp.loss_fn.discriminator_loss import DenseDiscriminator
from utils.img import transform
from utils.lmdb import LMDBImageDataset, chexpert_indexer
from utils.utils import repeat


class Coach:
    def __init__(self):
        ckpt = torch.load(str(CONFIG.STYLEGAN_CKPT))
        self.generator = Generator(CONFIG.RESOLUTION).to("cuda")
        self.discriminator = DenseDiscriminator().to("cuda")

        self.generator.load_state_dict(ckpt["g_ema"])
        self.test_batch = 200
        self.loss = nn.CrossEntropyLoss().to("cuda")
        self.train_labels = (
            Tensor(
                [
                    *[1 for _ in range(CONFIG.STYLEGAN_BATCH)],
                    *[0 for _ in range(CONFIG.STYLEGAN_BATCH)],
                ]
            )
            .long()
            .to("cuda")
        )
        self.test_labels = (
            Tensor(
                [
                    *[1 for _ in range(self.test_batch)],
                    *[0 for _ in range(self.test_batch)],
                ]
            )
            .long()
            .to("cuda")
        )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=CONFIG.STYLEGAN_LR,
        )

        # Initialize dataset
        self.train_dataset = LMDBImageDataset(
            CONFIG.CHEXPERT_TRAIN_LMDB, chexpert_indexer, transform
        )
        self.test_dataset = LMDBImageDataset(
            CONFIG.CHEXPERT_TEST_LMDB, chexpert_indexer, transform, length=300
        )

        self.train_dataloader = repeat(
            DataLoader(
                self.train_dataset,
                batch_size=CONFIG.STYLEGAN_BATCH,
                shuffle=True,
                num_workers=2,
                prefetch_factor=CONFIG.STYLEGAN_BATCH,
                drop_last=True,
            )
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch,
            shuffle=False,
            num_workers=2,
            prefetch_factor=self.test_batch,
            drop_last=True,
        )

        # Initialize logger
        self.logger = SummaryWriter(log_dir=str(CONFIG.OUTPUT_DIR / "logs"))

        # Initialize checkpoint dir
        self.checkpoint_dir = CONFIG.OUTPUT_DIR / "checkpoint"

        self.start_iter = 0
        self.fake_imgs: List[Tensor] = []

    def train(self):
        self.generator.eval()
        self.generator.requires_grad_(False)
        self.discriminator.train()
        self.discriminator.requires_grad_(True)

        for step in tqdm(
            range(0, CONFIG.STYLEGAN_ITER),
            initial=self.start_iter,
            total=CONFIG.STYLEGAN_ITER,
            dynamic_ncols=True,
            smoothing=0.01,
        ):
            real_img: Tensor = next(self.train_dataloader)[0].to("cuda")
            noise: Any = mixing_noise(
                CONFIG.STYLEGAN_BATCH, CONFIG.LATENT_DIM, CONFIG.STYLEGAN_MIXING, "cuda"
            )
            fake_img: Tensor = self.generator(noise)

            # Configure input and target images
            imgs = torch.cat([real_img, fake_img], dim=0)
            pred, _ = self.discriminator(imgs)
            loss, loss_dict = self.calc_loss(pred, self.train_labels)

            # Back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.log_metrics(step, loss_dict, prefix="train")

            # Validation related
            if step % 200 == 0:
                self.test(step)

            if step % 5000 == 0:
                torch.save(
                    {"d": self.discriminator.state_dict()},
                    CONFIG.OUTPUT_DIR / f"checkpoint/{str(step).zfill(6)}.pt",
                )

    def test(self, step: int):
        self.generator.eval()
        self.discriminator.eval()

        loss_dicts: List[Dict[str, float]] = []

        for idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                real_img = batch[0].to("cuda")

                if len(self.fake_imgs) - 1 < idx:
                    noise: Any = mixing_noise(
                        self.test_batch,
                        CONFIG.LATENT_DIM,
                        CONFIG.STYLEGAN_MIXING,
                        "cuda",
                    )
                    self.fake_imgs.append(self.generator(noise))

                fake_img = self.fake_imgs[idx]

                # Configure input and target images
                imgs = torch.cat(
                    [real_img.expand(-1, 3, -1, -1), fake_img.expand(-1, 3, -1, -1)],
                    dim=0,
                )
                pred, _ = self.discriminator(imgs)
                _, loss_dict = self.calc_loss(pred, self.test_labels)

            loss_dicts.append(loss_dict)

        loss_dict = self.agg_loss(loss_dicts)
        self.log_metrics(step, loss_dict, prefix="test")
        return loss_dict

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

    def calc_loss(self, pred: Tensor, targ: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        # pred: (N, C)
        # targ: N

        # Calculate loss
        loss = self.loss(pred, targ)
        assert targ.shape[0] % 2 == 0
        with torch.no_grad():
            # Calculate probabilities
            n_real = targ.shape[0] // 2

            # Calculate accuracy
            _, pred_class = torch.max(pred, dim=1)
            total = targ.size(0)
            correct = (pred_class == targ).sum().item()

            loss_dict = {
                "loss": float(loss),
                "real_score": float(F.softmax(pred[:n_real], dim=1)[:, 1].mean()),
                "fake_score": float(F.softmax(pred[n_real:], dim=1)[:, 1].mean()),
                "accuracy": float(correct / total),
            }
        return loss, loss_dict


def main():
    coach = Coach()
    coach.train()


if __name__ == "__main__":
    main()

from pathlib import Path

from utils.config import CONFIG, guard

guard()

from typing import Dict, List

import torch
from efficientnet_pytorch import EfficientNet
from stylegan2_torch import Discriminator
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import densenet121, resnet50
from tqdm import tqdm

from utils.dataset import BinaryImageDataset
from utils.utils import repeat


class Coach:
    def __init__(self):
        self.net = EfficientNet.from_name(
            CONFIG.EFF_ARCH, in_channels=1, num_classes=2, image_size=256
        ).to("cuda")

        self.net = resnet50(num_classes=2).to("cuda")
        # self.net = EfficientNet.from_pretrained(
        #     "efficientnet-b1", in_channels=1, num_classes=2, image_size=256
        # ).to("cuda")

        # self.net = Discriminator(CONFIG.RESOLUTION).to("cuda")

        # Utility functions
        self.loss = nn.CrossEntropyLoss().to("cuda")
        # self.loss = nn.BCEWithLogitsLoss().to("cuda")
        self.sigmoid = nn.Sigmoid().to("cuda")

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=CONFIG.EFF_LR,
        )

        # Initialize dataset
        self.train_dataset = BinaryImageDataset(
            [
                CONFIG.PROJECT_DIR / "input/data" / item
                for item in CONFIG.EFF_TRAIN_POSITIVE.split(",")
            ],
            [
                CONFIG.PROJECT_DIR / "input/data" / item
                for item in CONFIG.EFF_TRAIN_NEGATIVE.split(",")
            ],
        )
        self.test_dataset = BinaryImageDataset(
            [
                CONFIG.PROJECT_DIR / "input/data" / item
                for item in CONFIG.EFF_TEST_POSITIVE.split(",")
            ],
            [
                CONFIG.PROJECT_DIR / "input/data" / item
                for item in CONFIG.EFF_TEST_NEGATIVE.split(",")
            ],
        )

        self.train_dataloader = repeat(
            DataLoader(
                self.train_dataset,
                batch_size=CONFIG.EFF_BATCH_SIZE,
                shuffle=True,
                num_workers=1,
                prefetch_factor=CONFIG.EFF_BATCH_SIZE * 4,
                drop_last=True,
            )
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=20,
            shuffle=True,
            num_workers=1,
            prefetch_factor=20,
            drop_last=True,
        )

        # Initialize logger
        self.logger = SummaryWriter(log_dir=str(CONFIG.OUTPUT_DIR / "logs"))

        # Initialize checkpoint dir
        self.checkpoint_dir = CONFIG.OUTPUT_DIR / "checkpoint"

        self.start_iter = 0
        # if self.net.resumed:
        #     self.start_iter = int(os.path.splitext(CONFIG.PSP_CKPT.stem)[0])

    def train(self):
        self.net.train()
        self.net.requires_grad_(True)

        for step in tqdm(
            range(self.start_iter, CONFIG.EFF_ITER),
            initial=self.start_iter,
            total=CONFIG.EFF_ITER,
            dynamic_ncols=True,
            smoothing=0.01,
        ):

            # Configure input and target images
            img: Tensor
            targ: Tensor
            img, targ = next(self.train_dataloader)

            img = img.to("cuda").repeat(1, 3, 1, 1)
            # targ = targ.to("cuda").float()
            targ = targ.to("cuda")

            # pred = self.net(img).squeeze(1)
            pred = self.net(img)

            loss = self.loss(pred, targ)

            # Back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = self.acc(pred, targ)

            self.log_metrics(step, {"loss": float(loss), "acc": acc}, prefix="train")

            # Validation related
            if step % CONFIG.EFF_TEST_INTERVAL == 0:
                _ = self.test(step)

            if step % CONFIG.EFF_CKPT_INTERVAL == 0:
                torch.save(
                    {
                        "net": self.net.state_dict(),
                        "optim": self.optimizer.state_dict(),
                        "config": CONFIG,
                    },
                    CONFIG.OUTPUT_DIR / f"checkpoint/{str(step).zfill(6)}.pt",
                )

    def test(self, step: int):
        self.net.eval()

        loss_dicts: List[Dict[str, float]] = []

        for _, (img, targ) in enumerate(self.test_dataloader):
            img: Tensor
            targ: Tensor
            img = img.to("cuda").repeat(1, 3, 1, 1)
            # targ = targ.to("cuda").float()
            targ = targ.to("cuda")

            with torch.no_grad():

                # pred = self.net(img).squeeze(1)
                pred = self.net(img)

                loss = self.loss(pred, targ)
                acc = self.acc(pred, targ)
            loss_dicts.append({"loss": float(loss), "acc": acc})

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

    def acc(self, pred: Tensor, targ: Tensor) -> float:
        with torch.no_grad():
            _, pred = torch.max(pred, dim=1)
            # pred = (self.sigmoid(pred) > 0.5).float()
            total = targ.size(0)
            correct = (pred == targ).sum().item()
            return correct / total


def main():
    coach = Coach()
    coach.train()


if __name__ == "__main__":
    main()

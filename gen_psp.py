from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils.config import CONFIG, guard
from utils.utils import repeat

guard()

from typing import Tuple

import cv2 as cv
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

from psp.pSp import pSp
from utils.img import transform
from utils.lmdb import LMDBImageDataset, covid_ct_indexer


class Coach:
    def __init__(self):
        ckpt = torch.load(str(CONFIG.PSP_CKPT))
        self.net = pSp(ckpt).to("cuda")

        # Initialize dataset
        self.train_dataset = LMDBImageDataset(
            CONFIG.COVID_19_TRAIN_LMDB,
            covid_ct_indexer,
            transform,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=CONFIG.PSP_BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            prefetch_factor=CONFIG.PSP_BATCH_SIZE,
            drop_last=True,
        )

        self.test_dataset = LMDBImageDataset(
            CONFIG.COVID_19_TEST_LMDB,
            covid_ct_indexer,
            transform,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=CONFIG.PSP_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            prefetch_factor=CONFIG.PSP_BATCH_SIZE,
            drop_last=True,
        )

        self.to_pil = ToPILImage("L")

    def gen(self, loader: DataLoader, output_dir: Path):
        self.net.eval()

        with tqdm(
            loader,
            total=len(loader) * CONFIG.PSP_BATCH_SIZE,
            dynamic_ncols=True,
            smoothing=0.01,
        ) as pbar:

            for idx, batch in enumerate(loader):
                raw, loc, drr, bone = [item.to("cuda") for item in batch]
                with torch.no_grad():

                    img_in, img_out, img_style, w_plus = self.forward_pass((raw, loc))

                # save_image(
                #     torch.concat(
                #         [raw, img_out],
                #         dim=3,
                #     ),
                #     str(output_dir / f"{str(idx).zfill(6)}.png"),
                #     nrow=1,
                #     normalize=True,
                #     value_range=(-1, 1),
                # )

                for i in range(img_out.shape[0]):

                    img_save = img_out[i][0]
                    img_save.clamp_(min=-1, max=1)
                    img_save.sub_(-1).div_(2)
                    ndarr = (
                        img_save.mul(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        # .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                    # im = Image.fromarray(ndarr)
                    # im.save(
                    #     str(
                    #         output_dir
                    #         / f"{str(idx * CONFIG.PSP_BATCH_SIZE + i).zfill(6)}.png"
                    #     )
                    # )
                    cv.imwrite(
                        str(
                            output_dir
                            / f"{str(idx * CONFIG.PSP_BATCH_SIZE + i).zfill(6)}.png"
                        ),
                        ndarr,
                    )
                    # self.to_pil(((img_out[i] + 1) / 2 * 255).squeeze(0).cpu()).save(
                    #     str(
                    #         output_dir
                    #         / f"{str(idx * CONFIG.PSP_BATCH_SIZE + i).zfill(6)}.png"
                    #     )
                    # )
                    # save_image(
                    #     img_out[i],
                    #     str(
                    #         output_dir
                    #         / f"{str(idx * CONFIG.PSP_BATCH_SIZE + i).zfill(6)}.png"
                    #     ),
                    #     nrow=1,
                    #     normalize=True,
                    #     value_range=(-1, 1),
                    # )
                    pbar.update(1)

    def forward_pass(
        self, tensors: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Concat image along channel direction
        img_in = torch.cat([img for img in tensors], dim=1)

        img_out: Tensor
        img_style: Tensor
        w_plus: Tensor
        img_out, img_style, w_plus = self.net(img_in)
        return img_in, img_out, img_style, w_plus


def main():
    coach = Coach()

    train_output_dir = CONFIG.OUTPUT_DIR / "covid_ct_gen_train"
    train_output_dir.mkdir(parents=True, exist_ok=True)
    coach.gen(coach.train_dataloader, train_output_dir)

    test_output_dir = CONFIG.OUTPUT_DIR / "covid_ct_gen_test"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    coach.gen(coach.test_dataloader, test_output_dir)


if __name__ == "__main__":
    main()

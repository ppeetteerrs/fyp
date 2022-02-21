import torch
from numpy import number
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from utils.utils import repeat


class Number(Dataset[Tensor]):
    def __init__(self) -> None:
        self.number = torch.randint(0, 1000, (5,))
        super().__init__()

    def __len__(self):
        return 5

    def __getitem__(self, index: int) -> Tensor:
        return self.number[index]


dataloader = repeat(
    DataLoader(
        Number(),
        batch_size=1,
        shuffle=True,
    )
)

for i in range(15):
    print(next(dataloader))

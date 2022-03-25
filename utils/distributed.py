"""PyTorch distributed training utilities"""

from typing import Dict, Tuple

import torch
from torch import distributed
from torch.distributed import init_process_group
from torch.functional import Tensor


def setup_distributed() -> Tuple[int, int]:
    """
    Sets up torch.distributed

    Returns:
        Tuple[int, int]: Local rank, world size
    """
    init_process_group(backend="nccl", init_method="env://")
    assert distributed.is_available(), "torch.distributed not available"
    assert distributed.is_initialized(), "torch.distributed not initialized"
    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    distributed.barrier()
    return local_rank, world_size


def reduce_sum(tensor: Tensor) -> Tensor:

    tensor = tensor.clone()
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)

    return tensor


def reduce_loss_dict(loss_dict: Dict[str, Tensor]):
    world_size = distributed.get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        distributed.reduce(losses, dst=0)

        if distributed.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

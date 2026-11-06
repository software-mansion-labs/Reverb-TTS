import os
import torch
import torch.distributed as dist

from typing import Tuple


def world() -> Tuple[int, int]:
    """
    This process' rank, and how many there are.

    Reads what the launcher set, so a run started without one is simply a world
    of a single rank - which is how every helper here collapses to nothing.
    """

    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    """
    Whether this is the rank that logs, evaluates and writes checkpoints.
    """

    return world()[0] == 0


def setup() -> torch.device:
    """
    Joins the other ranks, and says which device this one owns.
    """

    rank, world_size = world()

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)

        return torch.device("cuda", rank)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def teardown() -> None:
    """
    Leaves the group, if there was one.
    """

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: float) -> float:
    """
    The average of a number across the ranks.

    Only what gets logged goes through here - gradients are averaged by the
    model wrapper itself.
    """

    if world()[1] == 1:
        return value

    total = torch.tensor(value, device=torch.cuda.current_device())
    dist.all_reduce(total)

    return float(total) / world()[1]


def barrier() -> None:
    """
    Holds the ranks together, so one of them can work alone for a while.
    """

    if world()[1] > 1:
        dist.barrier()

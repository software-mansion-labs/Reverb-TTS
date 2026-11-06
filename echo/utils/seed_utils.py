import numpy as np
import random
import torch


def seed_of(*keys: int) -> int:
    """
    Folds a draw's place in a run into an unique seed of its own.
    """

    seed = 0
    for key in keys:
        seed = (seed * 1_000_003 + key) % (2**31)

    return seed


def seed_everything(seed: int) -> None:
    """
    Pins every stream a run draws from.

    CUDA follows 'torch.manual_seed' on its own, and MPS does not, so it is
    seeded by hand.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """
    Gives every data loader worker its own stream.

    Torch seeds itself per worker, but the preprocessor draws its prefix split
    and its dropped prefixes from 'random', which torch leaves alone - so
    without this the workers would replay one another's noise.
    """

    seed = torch.initial_seed() % (2**31)

    random.seed(seed_of(seed, worker_id))
    np.random.seed(seed_of(seed, worker_id))

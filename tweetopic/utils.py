import numpy as np
from numba import njit


@njit
def _jitted_seed(seed: int):
    np.random.seed(seed)


def set_numba_seed(seed: int):
    np.random.seed(seed)
    _jitted_seed(seed)

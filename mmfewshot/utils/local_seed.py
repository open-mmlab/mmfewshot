from contextlib import contextmanager
from typing import Optional

import numpy as np


@contextmanager
def local_numpy_seed(seed: Optional[int] = None) -> None:
    """Run numpy codes with a local random seed.

    If seed is None, the default random state will be used.
    """
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

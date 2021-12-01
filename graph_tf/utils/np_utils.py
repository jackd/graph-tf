import contextlib

import numpy as np


@contextlib.contextmanager
def random_seed_context(seed: int):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)

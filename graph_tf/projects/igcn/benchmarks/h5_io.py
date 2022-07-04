import contextlib
import os
import tempfile

import h5py
import numpy as np
import tqdm

n = 10000
chunk_size = 16
n_test = chunk_size * 10


@contextlib.contextmanager
def tempfile_context():
    path = tempfile.mktemp()
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


desc_width = len(f"Writing columns (chunk_size={chunk_size})...")

print(f"h5_io benchmark, n = {n}")
with tempfile_context() as path:
    fp = h5py.File(path, mode="a")
    data = fp.create_dataset("data", shape=(n, n), dtype=np.float32)
    for i in tqdm.trange(n_test, desc="Writing rows...".ljust(desc_width)):
        data[i] = np.random.normal(size=(n,)).astype(np.float32)
    for i in tqdm.trange(n_test, desc="Writing columns...".ljust(desc_width)):
        data[:, i] = np.random.normal(size=(n,)).astype(np.float32)
    with tqdm.tqdm(
        desc=f"Writing columns (chunk_size={chunk_size})...".ljust(desc_width),
        total=n_test - n_test % chunk_size,
    ) as prog:
        for i in range(n_test // chunk_size):
            data[:, i * chunk_size : (i + 1) * chunk_size] = np.random.normal(
                size=(n, chunk_size)
            ).astype(np.float32)
            prog.update(chunk_size)
    for i in tqdm.trange(n_test, desc="Reading rows...".ljust(desc_width)):
        data[i]
    for i in tqdm.trange(n_test, desc="Reading columns...".ljust(desc_width)):
        data[:, i]
    with tqdm.tqdm(
        desc=f"Reading columns (chunk_size={chunk_size})...".ljust(desc_width),
        total=n_test - n_test % chunk_size,
    ) as prog:
        for i in range(n_test // chunk_size):
            data[:, i * chunk_size : (i + 1) * chunk_size]
            prog.update(chunk_size)
    r = np.arange(n_test, dtype=np.int64)
    with tqdm.tqdm(
        desc=f"Reading columns random indices".ljust(desc_width),
        total=n_test - n_test % chunk_size,
    ) as prog:
        for i in range(n_test // chunk_size):
            np.random.shuffle(r)
            data[:, np.sort(r[:chunk_size])]
            prog.update(chunk_size)

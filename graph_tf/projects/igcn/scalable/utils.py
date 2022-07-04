import contextlib
import os
import shutil
import typing as tp

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tqdm

DEFAULT_TOL: float = 1e-5
DEFAULT_MAXITER: tp.Optional[int] = None


def to_shifted_laplacian(
    adjacency: sp.spmatrix, epsilon: float, identity_coeff: float = 1.0
) -> sp.spmatrix:
    adjacency = adjacency.tocoo(copy=False)
    n = adjacency.shape[0]
    d = np.squeeze(np.asarray(adjacency.sum(axis=1)), axis=1)
    d = 1 / np.sqrt(d)
    data = (1 - epsilon) * d[adjacency.row] * d[adjacency.col]
    return sp.eye(n) * identity_coeff - sp.coo_matrix(
        (data, (adjacency.row, adjacency.col))
    )


def shifted_laplacian_solver(
    adjacency: sp.spmatrix,
    epsilon: float,
    maxiter: tp.Optional[int] = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
) -> tp.Callable[[np.ndarray], np.ndarray]:
    L = to_shifted_laplacian(adjacency, epsilon=epsilon)
    L = L.tocsr()

    def solve(rhs: np.ndarray) -> np.ndarray:
        """Shifted Laplacian inverse computed with conjugate gradient."""
        x, info = la.cg(L, rhs, maxiter=maxiter, tol=tol)
        del info
        return x

    return solve


@contextlib.contextmanager
def remove_on_exception_context(
    path: str,
    exception_cls: tp.Union[type, tp.Tuple[type, ...]] = (Exception, KeyboardInterrupt),
):
    try:
        yield
    except exception_cls:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        raise


def create_transpose(data: np.ndarray, dst: h5py.Group, key: str, block_size: int):
    n, m = data.shape
    out = dst.create_dataset(key, shape=(m, n), dtype=data.dtype)
    try:
        with tqdm.tqdm(total=m, desc=f"Creating transpose dataset at {key}") as prog:
            for i in range(0, m, block_size):
                bs = min(block_size, m - i)
                out[i : i + bs] = data[:, i : i + bs].T
                prog.update(bs)
    except (Exception, KeyboardInterrupt):
        del dst[key]
        raise


def assert_exists(path: str):
    assert os.path.exists(path), f"Nothing exists at {path}"

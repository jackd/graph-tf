import typing as tp

import google_benchmark as benchmark
import numba as nb
import numpy as np

seed = 0
n = 1024
m = 10_000
dtype = np.float32


def get_x(seed: int, n: int, m: int, dtype: np.dtype):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, m)).astype(dtype)


@nb.njit()
def xxt_v0(x: np.ndarray) -> np.ndarray:
    return x @ x.T


@nb.njit()
def xxt_v1(x: np.ndarray) -> np.ndarray:
    n, m = x.shape
    out = np.zeros((n, n), dtype=x.dtype)
    for i in range(m):
        xi = x[:, i]
        out += np.outer(xi, xi)
    return out


@nb.njit(parallel=True)
def xxt_v2(x: np.ndarray) -> np.ndarray:
    n, m = x.shape
    out = np.empty((n, n), dtype=x.dtype)
    for i in nb.prange(n):
        xi = x[i]  # [m]
        out[i] = x @ xi
    return out


def run_benchmark(fn: tp.Callable, state):
    x = get_x(seed, n, m, dtype)

    # warmup
    fn(x)

    while state:
        fn(x)


@benchmark.register
def numpy_impl(state):
    def f(x):
        return x @ x.T

    run_benchmark(f, state)


@benchmark.register
def xxt_v0_impl(state):
    run_benchmark(xxt_v0, state)


@benchmark.register
def xxt_v1_impl(state):
    run_benchmark(xxt_v1, state)


@benchmark.register
def xxt_v2_impl(state):
    run_benchmark(xxt_v2, state)


if __name__ == "__main__":
    benchmark.main()

from typing import Optional

import jax.numpy as jnp
from jju.linalg import custom_gradients as cg
from jju.linalg import subspace_iteration as si
from jju.linalg import utils
from jju.sparse import coo


def chebyshev_subspace_iteration_coo(
    data: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    sized: jnp.ndarray,
    v0: jnp.ndarray,
    tol: Optional[float] = None,
    max_iters: int = 1000,
    scale: float = 1.0,
    order: int = 8,
):
    a = coo.matmul_fun(data, row, col, sized)
    w, v, info = si.chebyshev_subspace_iteration(
        order, scale, a, v0, tol=tol, max_iters=max_iters
    )
    del info
    v = utils.standardize_signs(v)
    return w, v


def eigh_partial_rev_coo(
    grad_w: jnp.ndarray,
    grad_v: jnp.ndarray,
    w: jnp.ndarray,
    v: jnp.ndarray,
    l0: jnp.ndarray,
    data: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    sized: jnp.ndarray,
    tol: float = 1e-5,
):
    """
    Args:
        grad_w: [k] gradient w.r.t eigenvalues
        grad_v: [m, k] gradient w.r.t eigenvectors
        w: [k] eigenvalues.
        v: [m, k] eigenvectors.
        l0: initial solution to least squares problem.
        data, row, col, sized: coo formatted [m, m] matrix.
        seed: used to initialize conjugate gradient solution.

    Returns:
        grad_data: gradient of `data` input, same `shape` and `dtype`.
        l: solution to least squares problem.
    """
    outer_impl = coo.masked_outer_fun(row, col)
    a = coo.matmul_fun(data, row, col, sized)
    grad_data, l0 = cg.eigh_partial_rev(
        grad_w, grad_v, w, v, l0, a, outer_impl, tol=tol
    )
    return grad_data, l0

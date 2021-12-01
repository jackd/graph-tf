from typing import Optional, Union

import jax
import tensorflow as tf
from jax.experimental import jax2tf

from graph_tf.projects.spag import jax_ops


def chebyshev_subspace_iteration_coo(
    data: tf.Tensor,
    row: tf.Tensor,
    col: tf.Tensor,
    v0: Union[tf.Variable, tf.Variable],
    l0: Union[tf.Variable, tf.Variable],
    tol: Optional[float] = None,
    max_iters: int = 1000,
    grad_tol: float = 1e-5,
    order: int = 8,
    scale: float = 1.0,
    start: int = 0,
    stop: Optional[int] = None,
):
    fwd = jax2tf.convert(
        jax.jit(
            jax.tree_util.Partial(
                jax_ops.chebyshev_subspace_iteration_coo,
                scale=scale,
                order=order,
                tol=tol,
                max_iters=max_iters,
            )
        ),
        with_gradient=False,
    )

    rev = jax2tf.convert(
        jax.jit(jax.tree_util.Partial(jax_ops.eigh_partial_rev_coo, tol=grad_tol)),
        with_gradient=False,
    )

    @tf.custom_gradient
    def f(data):
        m = v0.shape[0]
        sized = tf.zeros((m,), dtype=tf.bool)
        w, v = fwd(data, row, col, sized, v0)
        if isinstance(v0, tf.Variable):
            v0.assign(v)
        w = w[start:stop]
        v = v[:, start:stop]

        def grad_fun(grad_w, grad_v):
            grad_data, l = rev(grad_w, grad_v, w, v, l0, data, row, col, sized)
            if isinstance(l0, tf.Variable):
                l0.assign(l)
            return grad_data

        return (w, v), grad_fun

    return f(data)


def chebyshev_subspace_iteration_sparse(
    a: tf.SparseTensor,
    v0: Union[tf.Tensor, tf.Variable],
    l0: Union[tf.Tensor, tf.Variable],
    *,
    tol: Optional[float] = None,
    max_iters: int = 1000,
    sparse_impl: str = "coo",
    order: Optional[int] = None,
    scale: float = 1.0,
):
    if v0.shape[1] is None:
        raise ValueError("v0.shape[1] must be statically known.")
    if l0.shape != v0.shape:
        raise ValueError(
            f"`v0.shape` must be the same as `l0.shape`, but {v0.shape} != {l0.shape}"
        )
    if order is None:
        order = 2 * v0.shape[1]
    kwargs = dict(tol=tol, max_iters=max_iters, order=order, scale=scale)
    data = a.values
    row, col = tf.unstack(a.indices, axis=-1)
    col = tf.cast(col, tf.int32)
    if sparse_impl == "coo":
        row = tf.cast(row, tf.int32)
        return chebyshev_subspace_iteration_coo(data, row, col, v0, l0, **kwargs)
    if sparse_impl == "csr":
        # TODO
        raise NotImplementedError("TODO")
    raise NotImplementedError(f"sparse_impl must be 'coo' or 'csr', got {sparse_impl}")

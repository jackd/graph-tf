"""Tensorflow port of `jju.linalg.custom_gradients`."""
from functools import partial
from typing import Callable

import tensorflow as tf


def project(x, v):
    x.shape.assert_has_rank(1)
    v.shape.assert_has_rank(1)
    return x - tf.matmul(
        tf.expand_dims(tf.math.conj(v), axis=1).conj(),
        tf.matmul(tf.expand_dims(v, axis=0), x),
    )


def projector(v):
    return partial(project, v=v)


def eigh_partial_rev(
    grad_w: tf.Tensor,
    grad_v: tf.Tensor,
    w: tf.Tensor,
    v: tf.Tensor,
    a: tf.linalg.LinearOperator,
    x0: tf.Tensor,
    outer_impl: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    tol: float = 1e-5,
):
    """
    Args:
        grad_w: [k] gradient w.r.t eigenvalues
        grad_v: [m, k] gradient w.r.t eigenvectors
        w: [k] eigenvalues
        v: [m, k] eigenvectors
        a: matrix LinaerOperator.
        x0: [m, k] initial solution to (A - w[i]I)x[i] = Proj(grad_v[:, i])
        tol: tolerance used in `jnp.linalg.cg`

    Returns:
        grad_a: [m, m]
        x0: [m, k]
    """
    grad_As = []

    grad_ws = tf.unstack(grad_w)
    vs = tf.unstack(v, axis=1)
    for (grad_wi, vi) in zip(grad_ws, vs):
        grad_As.append(grad_wi * outer_impl(vi.conj(), vi))

    if grad_v is not None:
        # Add eigenvector part only if non-zero backward signal is present.
        # This can avoid NaN results for degenerate cases if the function
        # depends on the eigenvalues only.

        def f_inner(grad_vi, wi, vi, x0i):
            def if_any():

                # Amat = (a - wi * jnp.eye(m, dtype=a.dtype)).T
                Amat = lambda x: (a(x.conj())).conj() - wi * x

                # Projection operator on space orthogonal to v
                P = projector(vi)

                # Find a solution lambda_0 using conjugate gradient
                (l0, _) = tf.linalg.experimental.conjugate_gradient(
                    Amat, P(grad_vi), x=P(x0i), tol=tol
                )
                # (l0, _) = jax.scipy.sparse.linalg.gmres(Amat, P(grad_vi), x0=P(x0i))
                # l0 = jax.numpy.linalg.lstsq(Amat, P(grad_vi))[0]
                # Project to correct for round-off errors
                # print(Amat(l0) - P(grad_vi))
                l0 = P(l0)
                return -outer_impl(l0, vi), l0

            def if_none():
                return tf.zeros_like(grad_As[0]), x0i

            return tf.cond(tf.reduce_any(grad_vi), if_any, if_none)

        x0s = []
        grad_vs = tf.unstack(grad_v, axis=1)
        ws = tf.unstack(w, axis=0)
        x0s = tf.unstack(x0, axis=-1)
        for grad_vi, wi, vi, x0i in zip(grad_vs, ws, vs, x0s):
            grad_ai, xi = f_inner(grad_vi, wi, vi, x0i)
            grad_As.append(grad_ai)
            x0s.append(xi)
        x0 = tf.stack(x0s, axis=0)
        # TODO: revert the above back to using vmap
        # it seems to cause issues with jax2tf.convert
        # grad_a, x0 = jax.vmap(f_inner, in_axes=(1, 0, 1, 1), out_axes=(0, 1))(
        #     grad_v, w, v, x0
        # )
        # grad_As.append(grad_a.sum(0))
    return tf.add_n(grad_As), x0

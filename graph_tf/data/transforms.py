import functools
import typing as tp

import gin
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf
import tqdm
from tflo.extras import LinearOperatorSparseMatrix

from graph_tf.utils import scipy_utils
from graph_tf.utils.graph_utils import (
    approx_effective_resistance_z,
    laplacian,
    signed_incidence,
    tril,
    tril_indices,
)
from graph_tf.utils.ops import normalize_sparse
from graph_tf.utils.random_utils import as_rng
from graph_tf.utils.type_checks import is_sparse_tensor

register = functools.partial(gin.register, module="gtf.data.transforms")
configurable = functools.partial(gin.configurable, module="gtf.data.transforms")

# back compatible support
register(laplacian)
register(signed_incidence)
register(tril)


@register
def transformed(base, transforms: tp.Union[tp.Callable, tp.Iterable[tp.Callable]]):
    if transforms is None:
        return base
    if callable(transforms):
        return transforms(base)
    for transform in transforms:
        base = transform(base)
    return base


@register
def add_identity(x: tf.SparseTensor) -> tf.SparseTensor:
    if isinstance(x, tf.SparseTensor):
        return tf.sparse.add(x, tf.sparse.eye(x.shape[0], dtype=x.dtype))
    assert tf.is_tensor(x)
    return x + tf.eye(tf.shape(x)[0])


@register
def normalize_symmetric(x: tp.Union[tf.Tensor, tf.SparseTensor]) -> tf.SparseTensor:
    if is_sparse_tensor(x):
        return normalize_sparse(x, symmetric=True)
    d = tf.reduce_sum(tf.abs(x), axis=1)
    d = tf.where(d <= 0, tf.zeros_like(d), tf.math.rsqrt(d))
    return x * d * tf.expand_dims(d, -1)


@register
def normalize_asymmetric(x: tp.Union[tf.Tensor, tf.SparseTensor]) -> tf.SparseTensor:
    if is_sparse_tensor(x):
        return normalize_sparse(x, symmetric=False)
    D = tf.reduce_sum(x, axis=1, keepdims=True)
    D = tf.where(D == 0, tf.zeros_like(D), tf.math.reciprocal(D))
    return x * D


@register
def sparsify(
    adj: tf.SparseTensor,
    *,
    epsilon: float = 0.3,
    matrix_conc_const: float = 4.0,
    jl_factor: float = 4.0,
    rng: tp.Union[int, tf.random.Generator] = 0,
    **cg_kwargs,
) -> tf.SparseTensor:
    rng: tf.random.Generator = as_rng(rng)
    Z = approx_effective_resistance_z(adj, jl_factor=jl_factor, rng=rng, **cg_kwargs)
    i, j, v = tril_indices(adj, return_values=True)
    m = tf.shape(i, tf.int64)[0]

    n = tf.cast(adj.dense_shape[0], tf.float32)

    Zi = tf.gather(Z, i, axis=0)
    Zj = tf.gather(Z, j, axis=0)
    prs = tf.reduce_sum(tf.math.squared_difference(Zi, Zj), axis=1)
    prs = prs * matrix_conc_const * tf.math.log(n) / (epsilon**2) * v
    prs = tf.minimum(prs, tf.ones_like(prs))
    print(prs)
    mask = rng.uniform((m,)) < v * prs
    i, j, v, pr = (tf.boolean_mask(vec, mask) for vec in (i, j, v, prs))
    indices = tf.stack((i, j), axis=1)
    tril = tf.SparseTensor(indices, v / pr, adj.dense_shape)
    triu = tf.sparse.transpose(tril)
    adj = tf.sparse.add(tril, triu)
    adj = tf.sparse.reorder(adj)  # pylint: disable=no-value-for-parameter
    return adj


@configurable
def normalized_laplacian(
    x: tf.SparseTensor, symmetric: bool = True, shift: float = 0.0
) -> tf.SparseTensor:
    d = tf.sparse.reduce_sum(x, axis=0)
    if symmetric:
        d = tf.math.rsqrt(d)
        row, col = tf.unstack(x.indices, axis=1)
        x = x.with_values(
            -x.values * tf.gather(d, row, axis=0) * tf.gather(d, col, axis=0)
        )
    else:
        x = x.with_values(-x.values / tf.gather(d, x.indices[:, 0], axis=0))
    return tf.sparse.add(
        tf.sparse.eye(x.dense_shape[0], dtype=x.dtype) * (1 + shift), x
    )


@configurable
def to_symmetric(x: tf.SparseTensor, half: bool = False) -> tf.SparseTensor:
    xt = tf.sparse.reorder(  # pylint: disable=no-value-for-parameter
        tf.sparse.transpose(x)
    )
    x = tf.sparse.add(x, xt)
    if half:
        x = x.with_values(x.values / 2)
    return x


@configurable
def remove_diag(x: tf.SparseTensor) -> tf.SparseTensor:
    row, col = tf.unstack(x.indices, axis=1)
    mask = row != col
    return tf.SparseTensor(
        tf.boolean_mask(x.indices, mask), tf.boolean_mask(x.values, mask), x.dense_shape
    )


@register
def actually_none(_: tp.Any):
    return None


@configurable
def to_format(x: tp.Union[tf.Tensor, tf.SparseTensor], fmt: str):
    if isinstance(x, tf.SparseTensor):
        if fmt == "dense":
            return tf.sparse.to_dense(x)
        assert fmt == "sparse"
        return x
    assert tf.is_tensor(x)
    if fmt == "dense":
        return x

    assert fmt == "sparse"
    return tf.sparse.from_dense(x)


@configurable
def row_normalize(x: tp.Union[tf.Tensor, tf.SparseTensor]):
    if isinstance(x, tf.Tensor):
        factor = tf.reduce_sum(x, axis=1, keepdims=True)
        factor = tf.where(factor == 0, tf.ones_like(factor), factor)
        return x / factor
    assert isinstance(x, tf.SparseTensor)
    row = x.indices[:, 0]
    factor = tf.math.segment_sum(x.values, row)
    return x.with_values(x.values / tf.gather(factor, row, axis=0))


def johnson_lindenstrauss_min_dim(n_samples: int, *, eps: float) -> int:
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return int((4 * np.log(n_samples) / denominator))


@register
def random_projection(
    features: tp.Union[tf.Tensor, tf.SparseTensor],
    k_or_eps: tp.Union[int, float],
    seed: tp.Optional[int] = None,
) -> tf.Tensor:
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)
    # num_nodes, n = features.shape
    n = features.shape[1]
    if isinstance(k_or_eps, int):
        k = k_or_eps
    else:
        assert isinstance(k_or_eps, float)
        k = johnson_lindenstrauss_min_dim(features.shape[1], eps=k_or_eps)
    if isinstance(features, tf.SparseTensor):
        R = rng.normal((n, k), stddev=1 / np.sqrt(k)).numpy()
        return tf.sparse.sparse_dense_matmul(features, R)
        # def project_col(col):
        #     return tf.squeeze(
        #         tf.sparse.sparse_dense_matmul(features, tf.expand_dims(col, axis=1)),
        #         axis=1,
        #     )

        # @tf.function
        # def project(R):
        #     return tf.map_fn(
        #         project_col,
        #         R,
        #         fn_output_signature=tf.TensorSpec(
        #             shape=(num_nodes,), dtype=features.dtype
        #         ),
        #     )

        # return tf.transpose(project(R))
    R = rng.normal((n, k), stddev=1 / np.sqrt(k))
    return tf.linalg.matmul(features, R)


@register
def page_rank_propagate(
    adj: tf.SparseTensor,
    x: tf.Tensor,
    epsilon: tp.Union[float, tp.Iterable[float]],
    symmetric: bool = True,
    tol: float = 1e-5,
    max_iter: int = 1000,
    parallel_iterations: tp.Optional[int] = None,
    device: str = "/cpu:0",
    show_progress: bool = True,
) -> tf.Tensor:
    adj.shape.assert_has_rank(2)
    dtype = adj.dtype
    assert dtype.is_floating
    num_nodes = adj.shape[0]
    x = tf.convert_to_tensor(x, dtype=dtype)
    x.shape.assert_has_rank(2)
    I = tf.sparse.eye(num_nodes, dtype=dtype)
    row_sum = tf.sparse.reduce_sum(adj, axis=1)
    x0 = tf.sqrt(row_sum) / tf.reduce_sum(row_sum).numpy()
    adj = normalize_sparse(adj, symmetric=symmetric)
    if isinstance(epsilon, float):
        epsilon = (epsilon,)
    else:
        assert len(epsilon) > 0

    out = []

    if symmetric:
        if show_progress:

            def solve(L: sp.spmatrix, x: tf.Tensor):
                x, info = la.cg(L, x, x0=x0, tol=tol, maxiter=max_iter)
                del info
                return x

        else:

            def solve(L: LinearOperatorSparseMatrix, x: tf.Tensor):
                iters, sol, *_ = tf.linalg.experimental.conjugate_gradient(
                    L, x, tol=tol, max_iter=max_iter, x=x0
                )
                del iters
                return sol

    else:
        raise NotImplementedError()

    for eps in epsilon:
        if eps == 1:
            out.append(x)
            continue
        assert 0 < eps < 1, eps
        L = tf.sparse.add(I, adj.with_values(adj.values * (eps - 1)))
        if parallel_iterations is None:
            parallel_iterations = x.shape[1]
        else:
            parallel_iterations = min(x.shape[1], parallel_iterations)

        if show_progress:
            L = scipy_utils.to_scipy(L)

            def solve_all(x: tf.Tensor):
                out = np.zeros(x.shape, dtype=x.dtype.as_numpy_dtype)
                for i in tqdm.trange(x.shape[1], desc="Computing page-rank vectors"):
                    out[:, i] = solve(L, x[:, i])  # pylint: disable=cell-var-from-loop
                return tf.convert_to_tensor(out, x.dtype)

        else:
            L = LinearOperatorSparseMatrix(
                L, is_self_adjoint=True, is_positive_definite=True
            )

            @tf.function
            def solve_all(x: tf.Tensor):
                # map only works on axis 0, hence the double transpose
                solT = tf.map_fn(
                    # lambda x: solve(L, x),
                    functools.partial(solve, L),  # pylint: disable=cell-var-from-loop
                    tf.transpose(x),
                    parallel_iterations=parallel_iterations,
                    fn_output_signature=tf.TensorSpec((num_nodes,), dtype=dtype),
                )
                return tf.transpose(solT)

        with tf.device(device):
            out.append(solve_all(x))
    if len(out) == 1:
        return out[0]
    return tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        out, axis=-1
    )

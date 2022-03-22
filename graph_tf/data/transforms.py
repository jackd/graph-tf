import functools
import typing as tp

import gin
import tensorflow as tf

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
    prs = prs * matrix_conc_const * tf.math.log(n) / (epsilon ** 2) * v
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

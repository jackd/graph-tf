import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.utils.ops import normalize_sparse

configurable = functools.partial(gin.configurable, module="gtf.data.transforms")


@configurable
def add_identity(x: tf.SparseTensor) -> tf.SparseTensor:
    if isinstance(x, tf.SparseTensor):
        return tf.sparse.add(x, tf.sparse.eye(x.shape[0], dtype=x.dtype))
    assert tf.is_tensor(x)
    return x + tf.eye(x.shape[0])


@configurable
def normalize_symmetric(x: tf.SparseTensor) -> tf.SparseTensor:
    return normalize_sparse(x, symmetric=True)


@configurable
def normalize_asymmetric(x: tf.SparseTensor) -> tf.SparseTensor:
    assert isinstance(x, tf.SparseTensor)
    return normalize_sparse(x, symmetric=False)


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


@configurable
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

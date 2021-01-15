from typing import Sequence

import tensorflow as tf

from graph_tf.utils.ops import collect_sparse


def random_sparse(
    dense_shape: Sequence[int],
    nnz: int,
    rng: tf.random.Generator,
    dtype: tf.DType = tf.float32,
):
    values = rng.normal((nnz,), dtype=dtype)
    indices = tf.cast(
        rng.uniform((nnz, len(dense_shape)))
        * tf.constant(dense_shape, dtype=tf.float32),
        tf.int64,
    )
    st = tf.SparseTensor(indices, values, dense_shape)
    st = tf.sparse.reorder(st)  # pylint: disable=no-value-for-parameter
    st = collect_sparse(st)
    return st

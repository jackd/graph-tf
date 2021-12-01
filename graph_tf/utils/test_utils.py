from typing import Sequence

import tensorflow as tf


def random_sparse(
    dense_shape: Sequence[int],
    nnz: int,
    rng: tf.random.Generator,
    dtype: tf.DType = tf.float32,
) -> tf.SparseTensor:
    indices = random_sparse_indices(dense_shape, nnz, rng)
    values = rng.normal((tf.shape(indices)[0],), dtype=dtype)
    return tf.SparseTensor(indices, values, dense_shape)


def random_sparse_indices(
    dense_shape: Sequence[int], nnz: int, rng: tf.random.Generator,
) -> tf.SparseTensor:
    max_index = tf.cast(tf.reduce_prod(dense_shape), tf.int64)
    indices = rng.uniform((nnz,), maxval=max_index, dtype=tf.int64)
    indices, _ = tf.unique(indices)
    indices = tf.transpose(tf.unravel_index(tf.sort(indices), dense_shape), (1, 0))
    return indices


def diags(values: tf.Tensor) -> tf.SparseTensor:
    n = tf.shape(values, out_type=tf.int64)[0]
    indices = tf.range(n)
    indices = tf.tile(tf.expand_dims(indices, axis=1), (1, 2))
    return tf.SparseTensor(indices, values, (n, n))


def random_laplacian(
    size: int,
    nnz: int,
    rng: tf.random.Generator,
    normalize: bool = False,
    dtype: tf.DType = tf.float32,
    shift: float = 0.0,
):
    dense_shape = (size, size)
    indices = random_sparse_indices(dense_shape, nnz, rng)
    valid = tf.not_equal(indices[:, 0], indices[:, 1])
    indices = tf.boolean_mask(indices, valid)
    # values = rng.uniform((tf.shape(indices)[0],), dtype=dtype)
    values = tf.ones((tf.shape(indices)[0],), dtype=dtype)
    st = tf.SparseTensor(indices, values, dense_shape)
    # make symmetric
    st = tf.sparse.add(st, tf.sparse.transpose(st, (1, 0)))
    row, col = tf.unstack(st.indices, axis=1)
    values = st.values
    row_sum = tf.math.segment_sum(values, row)
    if normalize:
        d = row_sum ** -0.5
        values = values * tf.gather(d, row, axis=0) * tf.gather(d, col, axis=0)
        eye = tf.sparse.eye(size, dtype=dtype)
        laplacian = tf.sparse.add(
            eye.with_values(eye.values * (1 - shift)), st.with_values(-values)
        )
    else:
        laplacian = tf.sparse.add(diags(row_sum - shift), st.with_values(-st.values))
    return laplacian, row_sum


if __name__ == "__main__":
    indices = random_sparse_indices((10, 10), 25, tf.random.Generator.from_seed(0))
    print(indices.numpy())

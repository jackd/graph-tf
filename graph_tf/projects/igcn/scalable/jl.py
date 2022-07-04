import typing as tp

import numpy as np
import tensorflow as tf


def johnson_lindenstrauss_min_dim(n_samples: int, *, eps: float) -> int:
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return int((4 * np.log(n_samples) / denominator))


def stateless_gaussian_projection_matrix(
    k: int,
    n_samples: int,
    seed: tf.Tensor,
    *,
    dtype: tf.DType = tf.float32,
    transpose: bool = False,
) -> tf.Tensor:
    shape = (k, n_samples)
    if transpose:
        shape = shape[-1::-1]
    return tf.random.stateless_normal(
        shape=shape, stddev=1 / np.sqrt(k), dtype=dtype, seed=seed
    )


def stateless_gaussian_projection_vector(
    k: int, n_samples: int, seed: tf.Tensor, dtype: tf.DType = tf.float32
) -> tf.Tensor:
    return tf.random.stateless_normal(
        shape=(n_samples,), stddev=1 / np.sqrt(k), dtype=dtype, seed=seed
    )


def gaussian_projection_matrix(
    k: int,
    n_samples: int,
    *,
    dtype: tf.DType = tf.float32,
    rng: tp.Optional[tf.random.Generator] = None,
    transpose: bool = False,
) -> tf.Tensor:
    if rng is None:
        rng = tf.random.get_global_generator()
    shape = (k, n_samples)
    if transpose:
        shape = shape[-1::-1]
    return rng.normal(shape, stddev=1 / np.sqrt(k), dtype=dtype)

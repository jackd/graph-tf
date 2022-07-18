import numbers
import typing as tp

import tensorflow as tf


def as_rng(
    rng: tp.Union[numbers.Integral, tf.random.Generator] = 0
) -> tf.random.Generator:
    if isinstance(rng, numbers.Integral):
        return tf.random.Generator.from_seed(rng)
    if isinstance(rng, tf.random.Generator):
        return rng
    raise TypeError(f"rng must be integral or Generator, got {rng}")


def stateless_shuffle(x: tf.Tensor, seed: tf.Tensor):
    order = tf.argsort(tf.random.stateless_uniform(shape=(tf.shape(x))[:1], seed=seed))
    return tf.gather(x, order)


def stateless_perm(size: tp.Union[tf.Tensor, int], seed: tf.Tensor):
    """Stateless random permuatation of values in [0, size)."""
    return tf.argsort(tf.random.stateless_uniform(shape=(size,), seed=seed))

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

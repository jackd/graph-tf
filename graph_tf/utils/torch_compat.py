import functools
import typing as tp
from typing import Callable, List, Tuple

import gin
import tensorflow as tf

GradsAndVars = List[Tuple[tf.Tensor, tf.Variable]]
GradientTransformer = Callable[[GradsAndVars], GradsAndVars]

register = functools.partial(gin.register, module="gtf.utils")


def linear_bias_initializer(fan_in: int):
    limit = tf.pow(tf.convert_to_tensor(fan_in, tf.float32), -0.5)
    return tf.keras.initializers.random_uniform(-limit, limit)


def linear_kernel_initializer():
    return tf.keras.initializers.VarianceScaling(
        scale=1 / 3, mode="fan_in", distribution="uniform"
    )


def with_weight_decay(
    grads_and_vars: GradsAndVars, weight_decay: float
) -> GradsAndVars:
    """Add weight decay to gradients as done in pytoch."""
    if weight_decay:
        return [(g + weight_decay * v, v) for g, v in grads_and_vars]
    return grads_and_vars


@register
def with_transform(
    grads_and_vars: GradsAndVars,
    transform: tp.Callable[[tf.Tensor, tf.Variable], tf.Tensor],
) -> GradsAndVars:
    return [(transform(g, v), v) for g, v in grads_and_vars]


@register
def weight_decay_transformer(weight_decay: float) -> GradientTransformer:
    """Get a `gradient_transformer` for use with `tf.keras.optimizers.Optimizer`."""
    return functools.partial(with_weight_decay, weight_decay=weight_decay)


@register
def weight_decay_to_l2(weight_decay: float) -> float:
    l2_coeff = weight_decay / 2
    return l2_coeff


@register
def l2_to_weight_decay(l2: float) -> float:
    weight_decay_coeff = l2 * 2
    return weight_decay_coeff

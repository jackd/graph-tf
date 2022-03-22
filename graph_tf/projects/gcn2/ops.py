import typing as tp

import tensorflow as tf

from graph_tf.utils import ops


def graph_conv(
    adj: tp.Union[tf.Tensor, tf.SparseTensor],
    features: tf.Tensor,
    features0: tf.Tensor,
    kernel: tf.Tensor,
    alpha: float,
    beta: float,
    variant: bool,
) -> tf.Tensor:
    hi = ops.matmul(adj, features)
    if variant:
        support = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            [hi, features0], axis=1
        )
        r = (1 - alpha) * hi + alpha * features
    else:
        support = (1 - alpha) * hi + alpha * features0
        r = support
    output = beta * support @ kernel + (1 - beta) * r
    return output

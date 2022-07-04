import functools

import gin
import tensorflow as tf

from graph_tf.projects.igcn import ops

register = functools.partial(gin.register, module="gtf.igcn.metrics")


def quadratic_sparse_categorical_crossentropy(
    y_true: tf.Tensor, y_pred: tf.Tensor, from_logits: bool = True
):
    if y_true.dtype.is_floating:
        y_true = tf.cast(y_true, tf.int64)
    if y_true.shape.ndims == 2:
        y_true = tf.squeeze(y_true, axis=1)
    return ops.quadratic_sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits
    )


@register
class QuadraticSparseCategoricalCrossentropy(tf.keras.metrics.MeanMetricWrapper):
    def __init__(
        self,
        from_logits: bool = True,
        name: str = "quadratic_sparse_categorical_crossentropy",
    ):
        super().__init__(
            fn=quadratic_sparse_categorical_crossentropy,
            from_logits=from_logits,
            name=name,
        )
        self.from_logits = from_logits

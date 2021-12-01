import typing as tp

import gin
import tensorflow as tf


class WeightedMeanLossWrapper(tf.keras.losses.Loss):
    def __init__(self, name: str, fn: tp.Callable, **kwargs):
        self.fn = fn
        self.kwargs = kwargs
        super().__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.call(y_true, y_pred)
        if sample_weight is None:
            return tf.reduce_mean(loss)
        if sample_weight.shape.ndims == loss.shape.ndims + 1:
            sample_weight = tf.squeeze(sample_weight, axis=-1)
        return tf.reduce_sum(loss * sample_weight) / tf.reduce_mean(sample_weight)

    def call(self, y_true, y_pred):
        return self.fn(y_true, y_pred, **self.kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(fn=self.fn, **self.kwargs)
        return config


@gin.register(module="gtf.utils.losses")  # pylint: disable=too-few-public-methods
class WeightedMeanBinaryCrossentropy(WeightedMeanLossWrapper):
    def __init__(
        self,
        name: str = "weighted_mean_binary_crossentropy",
        from_logits: bool = False,
    ):
        super().__init__(
            name=name, fn=tf.keras.backend.binary_crossentropy, from_logits=from_logits
        )


@gin.register(module="gtf.utils.losses")
class WeightedMeanSparseCategoricalCrossentropy(WeightedMeanLossWrapper):
    def __init__(
        self,
        name: str = "weighted_mean_sparse_categorical_crossentropy",
        from_logits: bool = False,
    ):
        super().__init__(
            name=name,
            fn=tf.keras.backend.sparse_categorical_crossentropy,
            from_logits=from_logits,
        )

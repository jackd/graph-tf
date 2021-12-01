import typing as tp

import gin
import tensorflow as tf


class WeightedMeanMetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, name: str, fn: tp.Callable, **kwargs):
        self.fn = fn
        self.kwargs = kwargs
        super().__init__(name)
        self.weighted_sum = self.add_weight(
            "weighted_sum",
            shape=(),
            dtype=self.dtype,
            initializer=tf.keras.initializers.Zeros(),
        )
        self.total_weight = self.add_weight(
            "total_weight",
            shape=(),
            dtype=self.dtype,
            initializer=tf.keras.initializers.Zeros(),
        )

    def get_config(self):
        config = super().get_config()
        config.update(fn=self.fn, **self.kwargs)
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        value = self.fn(y_true, y_pred)
        if sample_weight is None:
            self.total_weight.add_assign(
                tf.cast(tf.reduce_prod(tf.shape(value)), self.dtype)
            )
            self.weighted_sum.add_assign(tf.reduce_sum(value))
        else:
            self.weighted_sum.add_assign(tf.reduce_sum(value * sample_weight))
            self.total_weight.add_assign(
                tf.cast(tf.reduce_sum(sample_weight), self.dtype)
            )

    def result(self):
        return self.weighted_sum / self.total_weight


@gin.register(module="gtf.utils.metrics")
class WeightedMeanSparseCategoricalCrossentropy(WeightedMeanMetricWrapper):
    def __init__(
        self,
        name: str = "weighted_mean_sparse_categorical_crossentropy",
        from_logits: bool = False,
        axis: int = -1,
    ):
        super().__init__(
            fn=tf.keras.backend.sparse_categorical_crossentropy,
            name=name,
            from_logits=from_logits,
            axis=axis,
        )

import functools
import typing as tp

import gin
import tensorflow as tf

register = functools.partial(gin.register, module="gtf.isgae")


@register
def mlp(
    input_spec: tf.TensorSpec,
    hidden_units: tp.Iterable[int],
    output_units: int,
    dropout_rate: float = 0.0,
    kernel_regularizer: tp.Optional[tf.keras.regularizers.Regularizer] = None,
    activation="relu",
    final_activation=None,
) -> tf.keras.Model:
    def dropout(x):
        return tf.keras.layers.Dropout(dropout_rate)(x) if dropout_rate else x

    kwargs = dict(kernel_regularizer=kernel_regularizer,)
    inp = tf.keras.Input(type_spec=input_spec)
    x = inp
    for u in hidden_units:
        x = dropout(x)
        x = tf.keras.layers.Dense(u, activation=activation, **kwargs)(x)
    x = dropout(x)
    x = tf.keras.layers.Dense(output_units, activation=final_activation, **kwargs)(x)
    return tf.keras.Model(inp, x)

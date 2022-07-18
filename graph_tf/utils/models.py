import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.utils.layers import SparseDropout
from graph_tf.utils.type_checks import is_sparse_tensor

register = functools.partial(gin.register, module="gtf.utils.models")


@register
def batch_norm(x, *args, **kwargs):
    layer = tf.keras.layers.BatchNormalization(*args, **kwargs)
    return layer(x)


@register
def dense(x, units: int, activation=None, **kwargs):
    layer = tf.keras.layers.Dense(units=units, activation=activation, **kwargs)
    return layer(x)


@register
def prelu(x: tf.Tensor, **kwargs) -> tf.Tensor:
    return tf.keras.layers.PReLU(**kwargs)(x)


@register
def mlp(
    input_spec: tp.Union[tf.TensorSpec, tf.Tensor],
    output_units: int,
    hidden_units: tp.Union[int, tp.Iterable[int]],
    activation="relu",
    final_activation=None,
    input_dropout_rate: tp.Optional[int] = None,
    dropout_rate: float = 0.0,
    normalization: tp.Optional[tp.Callable] = None,
    dense_fn: tp.Callable = dense,
    hack_input_spec: bool = False,
) -> tf.keras.Model:
    if input_dropout_rate is None:
        input_dropout_rate = dropout_rate
    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,)
    activation: tp.Callable = tf.keras.activations.get(activation)

    def dropout(x, dropout_rate=dropout_rate):
        if dropout_rate:
            shape = x.shape
            if is_sparse_tensor(x):
                x = SparseDropout(dropout_rate)(x)
            else:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
            x.set_shape(shape)
        return x

    if tf.is_tensor(input_spec):
        inp = input_spec
    else:
        if hack_input_spec:
            input_spec = tf.nest.map_structure(
                lambda s: tf.TensorSpec((None, *s.shape[1:]), dtype=s.dtype), input_spec
            )
        inp = tf.keras.Input(type_spec=input_spec)
    x = dropout(inp, input_dropout_rate)

    for u in hidden_units:
        x = dense_fn(x, units=u, activation=None)  # activation after normalization
        if normalization:
            x = normalization(x)
        x = activation(x)
        x = dropout(x)
    logits = dense_fn(x, units=output_units, activation=final_activation)
    return tf.keras.Model(inp, logits)

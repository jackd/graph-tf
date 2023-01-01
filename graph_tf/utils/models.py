import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.utils.layers import SparseDropout
from graph_tf.utils.type_checks import is_sparse_tensor

# from graph_tf.utils.torch_compat import (
#     linear_bias_initializer,
#     linear_kernel_initializer,
# )

register = functools.partial(gin.register, module="gtf.utils.models")


@register
def batch_norm(x, *args, **kwargs):
    layer = tf.keras.layers.BatchNormalization(*args, **kwargs)
    return layer(x)


@register
def dense(x, units: int, activation=None, **kwargs):
    # import numpy as np

    # kwargs["kernel_initializer"] = tf.keras.initializers.VarianceScaling(
    #     scale=np.sqrt(10.0), mode="fan_avg", distribution="uniform"
    # )  # HACK
    layer = tf.keras.layers.Dense(units=units, activation=activation, **kwargs)
    out = layer(x)
    # HACK
    # layer.kernel.assign(
    #     linear_kernel_initializer()(layer.kernel.shape, layer.kernel.dtype)
    # )
    # layer.bias.assign(
    #     linear_bias_initializer(x.shape[1])(layer.bias.shape, layer.bias.dtype)
    # )
    return out


@register
def prelu(x: tf.Tensor, **kwargs) -> tf.Tensor:
    return tf.keras.layers.PReLU(**kwargs)(x)


def dropout(x: tp.Union[tf.Tensor, tf.SparseTensor], dropout_rate: float):
    if dropout_rate:
        shape = x.shape
        if is_sparse_tensor(x):
            x = SparseDropout(dropout_rate)(x)
        else:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        x.set_shape(shape)
    return x


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
    final_dense_fn: tp.Optional[tp.Callable] = None,
    hack_input_spec: bool = False,
) -> tf.keras.Model:
    if final_dense_fn is None:
        final_dense_fn = dense_fn
    if input_dropout_rate is None:
        input_dropout_rate = dropout_rate
    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,)
    if activation == "prelu":
        activation = prelu
    else:
        activation: tp.Callable = tf.keras.activations.get(activation)

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
        # activation after normalization
        x = dense_fn(x, units=u, activation=None)
        if normalization:
            x = normalization(x)
        x = activation(x)
        x = dropout(x, dropout_rate)
    logits = final_dense_fn(x, units=output_units, activation=final_activation)
    return tf.keras.Model(inp, logits)

import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.projects.cnet.layers import Propagation
from graph_tf.utils.layers import SparseDropout
from graph_tf.utils.type_checks import is_sparse_tensor

register = functools.partial(gin.register, module="gtf.cnet")


# @register
# def base_model(
#     inputs_spec,
#     num_classes: int,
#     hidden_units: int = 256,
#     num_hidden_layers: int = 2,
#     dropout_rate: float = 0.0,
#     use_batch_norm: bool = True,
#     activation="relu",
#     final_activation=None,
#     momentum: float = 0.9,
#     l2_reg: tp.Optional[float] = None,
#     use_skip_connections: bool = True
# ):
#     activation = tf.keras.activations.get(activation)
#     kernel_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
#     inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
#     T, x = inputs

#     for _ in range(num_hidden_layers):
#         x0 = x
#         x = tf.keras.layers.Dense(
#             hidden_units, activation=None, kernel_regularizer=kernel_regularizer
#         )(x)
#         x = Propagation()((T, x))
#         if use_skip_connections:
#             if x0.shape[1] != hidden_units:
#                 x0 = tf.keras.layers.Dense(
#                   hidden_units, activation=None, kernel_regularizer=kernel_regularizer
#                 )(x0)
#             x = x + x0
#         if use_batch_norm:
#             x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
#         x = activation(x)
#         if dropout_rate:
#             x = tf.keras.layers.Dropout(dropout_rate)(x)

#     x = tf.keras.layers.Dense(
#         num_classes,
#         activation=final_activation,
#         kernel_regularizer=kernel_regularizer,
#     )(x)
#     model = tf.keras.Model(inputs, x)
#     return model


@register
def base_model(
    inputs_spec,
    num_classes: int,
    units: int = 16,
    hidden_layers: int = 1,
    activation="relu",
    dropout_rate: tp.Optional[float] = 0.5,
    l2_reg: float = 0,  # original uses 5e-4 with tf.nn.l2_loss
    linear_skip_connections: bool = True,
    final_activation=None,
) -> tf.keras.Model:
    activation = tf.keras.activations.get(activation)

    def dropout(x):
        if dropout_rate:
            if is_sparse_tensor(x):
                return SparseDropout(dropout_rate)(x)
            return tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    def propagate(x, adj, filters, activation=None, kernel_regularizer=None):
        x = dropout(x)
        kwargs = dict(kernel_regularizer=kernel_regularizer, use_bias=False)
        if linear_skip_connections:
            skip = x
            x = tf.keras.layers.Dense(filters, **kwargs)(x)
            x = Propagation()((adj, x))
            x = x + skip
        else:
            dense = tf.keras.layers.Dense(filters, **kwargs)
            prop = Propagation()
            if filters <= x.shape[-1]:
                x = prop((adj, dense(x)))
            else:
                x = dense(prop((adj, x)))
        if activation is not None:
            x = activation(x)
        return x

    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    T, x = inputs

    kernel_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    x = tf.keras.layers.Dense(
        units, activation=activation, kernel_regularizer=kernel_regularizer
    )(x)

    for _ in range(hidden_layers):
        x = propagate(
            x, T, units, activation=activation, kernel_regularizer=kernel_regularizer,
        )

    logits = tf.keras.layers.Dense(
        num_classes, kernel_regularizer=kernel_regularizer, activation=final_activation
    )(x)
    return tf.keras.Model(inputs, logits)

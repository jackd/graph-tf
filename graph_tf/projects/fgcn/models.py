import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.projects.fgcn.layers import FactorizedGraphConvolution
from graph_tf.utils.layers import SparseDropout
from graph_tf.utils.type_checks import is_sparse_tensor

register = functools.partial(gin.register, module="gtf.fgcn")


@gin.configurable(module="gtf.fgcn")
def fgcn(
    inputs_spec,
    num_classes: int,
    hidden_filters: tp.Union[int, tp.Iterable[int]] = 16,
    activation="relu",
    dropout_rate: tp.Optional[float] = 0.5,
    l2_reg: float = 0.0,  # original uses 5e-4 with tf.nn.l2_loss
    linear_skip_connections: bool = False,
    final_activation=None,
):
    activation = tf.keras.activations.get(activation)

    def dropout(x):
        if dropout_rate:
            if is_sparse_tensor(x):
                x = SparseDropout(dropout_rate)(x)
            else:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    def propagate(x, filters, activation=None, kernel_regularizer=None):
        x = dropout(x)
        kwargs = dict(kernel_regularizer=kernel_regularizer, use_bias=False)
        if linear_skip_connections:
            skip = tf.keras.layers.Dense(filters, **kwargs)
            x = FactorizedGraphConvolution(filters, **kwargs)((x, V))
            if activation is not None:
                x = activation(x + skip)
        else:
            layer = FactorizedGraphConvolution(filters, activation=activation, **kwargs)
            x = layer((x, V))
        return x

    inputs = x, V = tf.nest.map_structure(
        lambda s: tf.keras.Input(type_spec=s), inputs_spec
    )

    if isinstance(hidden_filters, int):
        hidden_filters = (hidden_filters,)

    for filters in hidden_filters:
        x = propagate(
            x,
            filters,
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg else None,
        )

    outputs = propagate(x, num_classes, activation=final_activation)
    return tf.keras.Model(inputs, outputs)

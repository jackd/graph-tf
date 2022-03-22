from typing import Iterable, Optional, Union

import gin
import tensorflow as tf

from graph_tf.projects.lfgcn import layers as lfgcn_layers
from graph_tf.utils.layers import SparseDense, SparseDropout
from graph_tf.utils.type_checks import is_sparse_tensor


@gin.configurable(module="gtf.lfgcn")
def lfgcn(
    inputs_spec,
    num_classes: int,
    hidden_filters: Union[int, Iterable[int]] = 16,
    activation="relu",
    rank: int = 64,
    dropout_rate: Optional[float] = 0.5,
    l2_reg: float = 0,  # original uses 5e-4 with tf.nn.l2_loss
    linear_skip_connections: bool = False,
    final_activation=None,
    reg_coeff: float = 1e-2,
    V_activation="relu",
):
    activation = tf.keras.activations.get(activation)

    def dropout(x):
        if dropout_rate:
            if is_sparse_tensor(x):
                x = SparseDropout(dropout_rate)(x)
            else:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    def propagate(x, adj, filters, activation=None, kernel_regularizer=None):
        x = dropout(x)
        V = (SparseDense if is_sparse_tensor(x) else tf.keras.layers.Dense)(
            rank, activation=V_activation
        )(x)
        kwargs = dict(
            kernel_regularizer=kernel_regularizer, use_bias=False, reg_coeff=reg_coeff
        )
        if linear_skip_connections:
            skip = tf.keras.layers.Dense(filters, **kwargs)
            x = lfgcn_layers.LearnedFactorizedGraphConvolution(filters, **kwargs)(
                [x, V, adj]
            )
            if activation is not None:
                x = activation(x + skip)
        else:
            x = lfgcn_layers.LearnedFactorizedGraphConvolution(
                filters, activation=activation, **kwargs
            )([x, V, adj])
        return x

    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    features, adjacency = inputs

    if isinstance(hidden_filters, int):
        hidden_filters = (hidden_filters,)

    x = features

    for filters in hidden_filters:
        x = propagate(
            x,
            adjacency,
            filters,
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg else None,
        )

    outputs = propagate(x, adjacency, num_classes, activation=final_activation)
    return tf.keras.Model(inputs, outputs)

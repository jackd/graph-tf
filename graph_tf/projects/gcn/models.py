from typing import Iterable, Optional, Union

import gin
import tensorflow as tf

from graph_tf.projects.gcn import layers as gcn_layers
from graph_tf.utils.layers import SparseDropout
from graph_tf.utils.ops import sparse_gather
from graph_tf.utils.type_checks import is_sparse_tensor


@gin.configurable(module="gtf.gcn")
def gcn(
    inputs_spec,
    num_classes: int,
    hidden_filters: Union[int, Iterable[int]] = 16,
    activation="relu",
    dropout_rate: Optional[float] = 0.5,
    l2_reg: float = 0,  # original uses 5e-4 with tf.nn.l2_loss
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

    def propagate(x, adj, filters, activation=None, kernel_regularizer=None):
        x = dropout(x)
        kwargs = dict(kernel_regularizer=kernel_regularizer, use_bias=False)
        if linear_skip_connections:
            skip = tf.keras.layers.Dense(filters, **kwargs)
            x = layer_fn(filters, **kwargs)([x, adj])
            if activation is not None:
                x = activation(x + skip)
        else:
            x = layer_fn(filters, activation=activation, **kwargs)([x, adj])
        return x

    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    if len(inputs) == 3:
        features, adjacency, ids = inputs
    else:
        features, adjacency = inputs
        ids = None
    layer_fn = (
        gcn_layers.MultiGraphConvolution
        if isinstance(adjacency, (list, tuple))
        else gcn_layers.GraphConvolution
    )

    if isinstance(hidden_filters, int):
        hidden_filters = (hidden_filters,)

    x = features
    kernel_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg else None

    for filters in hidden_filters:
        x = propagate(
            x,
            adjacency,
            filters,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

    if ids is not None:
        adjacency = tf.nest.map_structure(lambda a: sparse_gather(a, ids).st, adjacency)
    outputs = propagate(
        x,
        adjacency,
        num_classes,
        activation=final_activation,
        # kernel_regularizer=kernel_regularizer,
    )
    return tf.keras.Model(inputs, outputs)

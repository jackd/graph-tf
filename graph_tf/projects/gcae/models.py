import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.projects.gcn import layers as gcn_layers
from graph_tf.utils.layers import SparseDropout
from graph_tf.utils.type_checks import is_sparse_tensor

register = functools.partial(gin.register, module="gtf.gcae")


@register
def gcae(
    inputs_spec,
    embedding_dim: int = 16,
    hidden_filters: tp.Union[int, tp.Iterable[int]] = 32,
    activation="relu",
    dropout_rate: tp.Optional[float] = 0.5,
    l2_reg: float = 0.0,
    linear_skip_connections: bool = False,
    final_activation=None,
) -> tf.keras.Model:
    if isinstance(hidden_filters, int):
        hidden_filters = (hidden_filters,)
    elif hidden_filters is None:
        hidden_filters = ()

    def propagate(x, adj, filters, activation=None, kernel_regularizer=None):
        if dropout_rate:
            if is_sparse_tensor(x):
                x = SparseDropout(dropout_rate)(x)
            else:
                x = tf.keras.layers.Dropout(dropout_rate)(x)

        kwargs = dict(kernel_regularizer=kernel_regularizer, use_bias=False)
        if linear_skip_connections:
            skip = tf.keras.layers.Dense(filters, **kwargs)
            x = graph_conv_factory(filters, **kwargs)([x, adj])
            x = x + skip
            if activation is not None:
                x = activation(x)
        else:
            x = graph_conv_factory(filters, activation=activation, **kwargs)([x, adj])
        return x

    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    x, adjacency = inputs
    graph_conv_factory = (
        gcn_layers.MultiGraphConvolution
        if isinstance(adjacency, (list, tuple))
        else gcn_layers.GraphConvolution
    )

    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    for filters in hidden_filters:
        x = propagate(
            x, adjacency, filters, activation=activation, kernel_regularizer=reg,
        )
    x = propagate(
        x, adjacency, embedding_dim, activation=final_activation, kernel_regularizer=reg
    )
    preds = tf.matmul(x, x, transpose_b=True)
    preds_flat = tf.reshape(preds, (-1, 1))
    return tf.keras.Model(inputs, preds_flat)

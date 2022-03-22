import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.projects.stale_gcn.layers import Propagation
from graph_tf.utils.layers import SparseDropout
from graph_tf.utils.type_checks import is_sparse_tensor

register = functools.partial(gin.register, module="gtf.stale_gcn")


@register
def gcn(
    inputs_spec,
    num_classes: int,
    hidden_filters: tp.Union[int, tp.Iterable[int]] = 16,
    activation="relu",
    dropout_rate: tp.Optional[float] = 0.5,
    l2_reg: float = 0,  # original uses 5e-4 with tf.nn.l2_loss
    linear_skip_connections: bool = False,
    final_activation=None,
) -> tf.keras.Model:
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

    if isinstance(hidden_filters, int):
        hidden_filters = (hidden_filters,)

    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    x, adjacency = inputs

    kernel_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    for filters in hidden_filters:
        x = propagate(
            x,
            adjacency,
            filters,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

    logits = propagate(
        x,
        adjacency,
        num_classes,
        activation=final_activation,
        # kernel_regularizer=kernel_regularizer,
    )
    return tf.keras.Model(inputs, logits)


@register
def dagnn(
    inputs_spec,
    num_classes: int,
    hidden_size: int = 256,
    dropout_rate: float = 0.2,
    num_propagations: int = 16,
    l2_reg: float = 0.0,
):
    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    x, adjacency = inputs

    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    kwargs = dict(kernel_regularizer=reg, bias_regularizer=reg,)

    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(hidden_size, activation="relu", **kwargs,)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(num_classes, **kwargs)(x)
    terms = [x]
    for _ in range(num_propagations):
        x = Propagation()((adjacency, x))
        terms.append(x)
    logits = tf.add_n(terms)
    # logits = MultiPropagation(num_propagations)((adjacency, logits))
    model = tf.keras.Model(inputs, logits)
    return model

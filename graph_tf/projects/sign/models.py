from typing import Callable, Optional

import gin
import numpy as np
import tensorflow as tf

from graph_tf.data.single import (
    SemiSupervisedSingle,
    preprocess_adjacency,
    preprocess_node_features,
)
from graph_tf.utils import ops
from graph_tf.utils.type_specs import keras_input


@gin.configurable(module="gtf.sign")
def preprocess_single(
    data: SemiSupervisedSingle,
    num_propagations: int,
    features_fn: Callable = preprocess_node_features,
    adjacency_fn: Callable[
        [tf.Tensor, tf.Tensor], tf.SparseTensor
    ] = preprocess_adjacency,
    # reorthogonalize: bool = False,
):
    edges = data.edges
    node_features = data.node_features
    num_nodes = tf.shape(node_features, out_type=tf.int64)[0]
    node_features = features_fn(node_features)
    adjacency = adjacency_fn(edges, num_nodes)
    node_features = ops.krylov(adjacency, node_features, num_propagations)
    nf = np.prod(node_features.shape[1:])
    node_features = tf.reshape(node_features, (num_nodes, nf))

    def get_data(ids):
        return (
            tf.gather(node_features, ids, axis=0),
            tf.gather(data.labels, ids, axis=0),
        )

    return (
        get_data(data.train_ids),
        get_data(data.validation_ids),
        get_data(data.test_ids),
    )


@gin.configurable(module="gtf.sign")
def mlp(
    inputs_spec,
    num_classes: int,
    hidden_units: int = 256,
    num_hidden_layers: int = 2,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = True,
    activation="relu",
    final_activation=None,
    momentum: float = 0.9,
    l2_reg: Optional[float] = None,
):
    assert isinstance(inputs_spec, tf.TensorSpec)
    activation = tf.keras.activations.get(activation)
    kernel_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    inp = keras_input(inputs_spec, use_batch_size=False)
    x = inp

    for _ in range(num_hidden_layers):
        x = tf.keras.layers.Dense(
            hidden_units, activation=None, kernel_regularizer=kernel_regularizer
        )(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
        x = activation(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(
        num_classes, activation=final_activation, kernel_regularizer=kernel_regularizer
    )(x)
    model = tf.keras.Model(inp, x)
    return model

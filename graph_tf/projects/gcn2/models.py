import functools

import gin
import numpy as np
import tensorflow as tf

from graph_tf.projects.gcn2.layers import GraphConvolution

register = functools.partial(gin.register, module="gtf.gcn2")


@register
def gcn2(
    inputs_spec,
    num_classes: int,
    filters: int = 64,
    num_hidden_layers: int = 64,
    dropout_rate: float = 0.6,
    lam: float = 0.5,
    alpha: float = 0.1,
    variant: bool = False,
    activation="relu",
    conv_weight_decay: float = 0.0,
    dense_weight_decay: float = 0.0,
):
    activation = tf.keras.activations.get(activation)

    def dropout(x):
        if dropout_rate:
            return tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    def dense(x: tf.Tensor, filters: int, **kwargs) -> tf.Tensor:
        reg = tf.keras.regularizers.l2(dense_weight_decay)
        return tf.keras.layers.Dense(
            filters, kernel_regularizer=reg, bias_regularizer=reg, **kwargs
        )(x)

    def conv(x: tf.Tensor, **kwargs):
        reg = tf.keras.regularizers.l2(conv_weight_decay)
        return GraphConvolution(
            filters, kernel_regularizer=reg, bias_regularizer=reg, **kwargs
        )((adjacency, x, x0))

    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.Input(type_spec=spec), inputs_spec
    )
    x, adjacency = inputs
    x = dropout(x)
    x = dense(x, filters, activation=activation, name="linear_0")
    x0 = x
    for i in range(num_hidden_layers):
        x = dropout(x)
        x = conv(
            x,
            variant=variant,
            beta=np.log(lam / (i + 1) + 1),
            alpha=alpha,
            use_bias=False,
            activation=activation,
        )

    x = dropout(x)
    x = dense(x, num_classes, name="linear_1")
    return tf.keras.Model(inputs, x)

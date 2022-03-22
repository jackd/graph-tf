import functools
import typing as tp

import gin
import numpy as np
import tensorflow as tf

from graph_tf.projects.stale_gcn.layers import Propagation

register = functools.partial(gin.register, module="gtf.stale_gcn")


class Multiply(tf.keras.layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return a * b


class Add(tf.keras.layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return a + b


def add(a, b):
    return Add()((a, b))


def mul(a, b):
    return Multiply()((a, b))


class Scale(tf.keras.layers.Layer):
    def __init__(self, factor: float, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def get_config(self):
        config = super().get_config()
        config["factor"] = self.factor
        return config

    def call(self, inputs):
        return self.factor * inputs


def graph_conv(
    adj: tp.Union[tf.Tensor, tf.SparseTensor],
    features: tf.Tensor,
    features0: tf.Tensor,
    alpha: float,
    beta: float,
    variant: bool,
    weight_decay: float,
    activation: tp.Callable,
    dense_name: str,
):

    filters = features.shape[-1]
    assert features0.shape[-1] == filters
    hi = Propagation()((adj, features))
    if variant:
        support = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            [hi, features0], axis=1
        )
        r = add(Scale(1 - alpha)(hi), Scale(alpha)(features))
    else:
        support = add(Scale(1 - alpha)(hi), Scale(alpha)(features0))
        r = support

    dense = tf.keras.layers.Dense(
        filters,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1 / 3, mode="fan_out", distribution="uniform"
        ),
        name=dense_name,
    )
    output = add(Scale(beta)(dense(support)), Scale(1 - beta)(r))
    return activation(output)


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

    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.Input(type_spec=spec), inputs_spec
    )
    x, adjacency = inputs
    x = dropout(x)
    x = dense(x, filters, activation=activation, name="linear_0")
    x0 = x
    for i in range(num_hidden_layers):
        x = dropout(x)
        x = graph_conv(
            adjacency,
            x,
            x0,
            alpha=alpha,
            beta=np.log(lam / (i + 1) + 1),
            variant=variant,
            weight_decay=conv_weight_decay,
            activation=activation,
            dense_name=f"conv_dense_{i}",
        )

    x = dropout(x)
    x = dense(x, num_classes, name="linear_1")
    return tf.keras.Model(inputs, x)

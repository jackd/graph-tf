import typing as tp

import gin
import tensorflow as tf

from graph_tf.projects.sgcn import ops


@gin.configurable(module="gtf.sgcn")
def transform_kernel(
    w,
    output_size: int,
    *,
    l2_reg: float = 0.0,
    activation: tp.Callable = tf.nn.relu,
    hidden_units: tp.Union[int, tp.Iterable[int]] = (),
    dropout_rate: float = 0.5,
    final_activation: tp.Callable = tf.nn.relu,
    # final_activation: tp.Callable = tf.identity,
    # final_activation: tp.Callable = tf.nn.softplus,
):
    def dropout(x):
        return tf.keras.layers.Dropout(dropout_rate)(x)

    if isinstance(hidden_units, int):
        hidden_units = (hidden_units,)
    w = tf.expand_dims(w, axis=1)

    kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
    w = dropout(w)
    for u in hidden_units:
        w = tf.keras.layers.Dense(
            u, activation=activation, kernel_regularizer=kernel_regularizer,
        )(w)
        w = dropout(w)
    w = tf.keras.layers.Dense(
        output_size, kernel_regularizer=kernel_regularizer, activation=final_activation
    )(w)
    return w


@gin.configurable(module="gtf.sgcn")
def sgcn(
    inputs_spec,
    num_classes: int,
    hidden_filters: tp.Iterable[int] = (16,),
    dropout_rate: float = 0.5,
    activation: tp.Callable = tf.nn.relu,
    final_activation: tp.Optional[tp.Callable] = None,
    residual_connections: bool = False,
    kernel_transform_fn=transform_kernel,
    l2_reg: float = 0.0,
    separable: bool = True,
    depthwise_multiplier: int = 4,
) -> tf.keras.Model:
    kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
    activation = tf.keras.activations.get(activation)
    final_activation = tf.keras.activations.get(final_activation)

    def dropout(x):
        out = tf.keras.layers.Dropout(dropout_rate)(x)
        out.set_shape(x.shape)
        return out

    if separable:

        def conv(node_features, w, V, filters_out):
            kernel = kernel_transform_fn(
                w, node_features.shape[1] * depthwise_multiplier
            )
            x = ops.depthwise_spectral_graph_conv(node_features, kernel, V)
            return tf.keras.layers.Dense(
                filters_out, kernel_regularizer=kernel_regularizer
            )(x)

    else:

        def conv(node_features, w, V, filters_out):
            kernel = kernel_transform_fn(w, node_features.shape[1] * filters_out)
            kernel = tf.reshape(kernel, (-1, node_features.shape[1], filters_out))
            return ops.spectral_graph_conv(node_features, kernel, V)

    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    node_features, w, V = inputs
    hidden_filters = list(hidden_filters)
    if residual_connections:
        node_features = dropout(node_features)
        node_features = tf.keras.layers.Dense(hidden_filters[0], activation=activation)(
            node_features
        )
    for f in hidden_filters:
        n0 = node_features
        node_features = dropout(node_features)
        node_features = conv(node_features, w, V, f)
        node_features = activation(node_features)
        if residual_connections:
            node_features = node_features + n0
    node_features = dropout(node_features)
    logits = tf.keras.layers.Dense(num_classes, kernel_regularizer=kernel_regularizer)(
        node_features
    )
    logits = final_activation(logits)
    model = tf.keras.Model(inputs, logits)
    return model

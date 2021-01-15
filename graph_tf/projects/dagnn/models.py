import gin
import tensorflow as tf

from graph_tf.projects.dagnn.layers import GatedSum
from graph_tf.utils import torch_compat
from graph_tf.utils.layers import Krylov
from graph_tf.utils.type_specs import keras_input


@gin.configurable(module="gtf.dagnn")
def dagnn_citations(
    inputs_spec,
    num_classes: int,
    hidden_size: int = 256,
    dropout_rate: float = 0.2,
    num_propagations: int = 16,
    l2_reg: float = 0.0,
):
    inputs = tf.nest.map_structure(keras_input, inputs_spec)
    if len(inputs) == 3:
        x, a, ids = inputs
    else:
        x, a = inputs
        ids = None

    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    kwargs = dict(
        kernel_initializer=torch_compat.linear_kernel_initializer(),
        kernel_regularizer=reg,
        bias_regularizer=reg,
    )

    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(
        hidden_size,
        activation="relu",
        bias_initializer=torch_compat.linear_bias_initializer(x.shape[-1]),
        **kwargs,
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    logits = tf.keras.layers.Dense(
        num_classes,
        bias_initializer=torch_compat.linear_bias_initializer(x.shape[-1]),
        **kwargs,
    )(x)
    logits = Krylov(num_propagations)([a, logits])
    logits = GatedSum(
        bias_initializer=torch_compat.linear_bias_initializer(logits.shape[-1]),
        **kwargs,
    )(logits)
    if ids is not None:
        logits = tf.gather(logits, ids, axis=0)
    model = tf.keras.Model(inputs, logits)
    return model

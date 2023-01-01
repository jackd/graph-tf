import typing as tp

import gin
import tensorflow as tf

from graph_tf.projects.dagnn.layers import GatedSum
from graph_tf.utils.layers import Krylov
from graph_tf.utils.models import dropout


@gin.configurable(module="gtf.dagnn")
def dagnn_citations(
    inputs_spec,
    num_classes: int,
    hidden_size: int = 256,
    dropout_rate: float = 0.2,
    num_propagations: int = 16,
    l2_reg: float = 0.0,
    input_dropout_rate: tp.Optional[float] = None,
    normalization: tp.Optional[tp.Callable] = None,
    activation="relu",
    simplified: bool = False,
) -> tf.keras.Model:
    activation = tf.keras.activations.get(activation)
    if input_dropout_rate is None:
        input_dropout_rate = dropout_rate
    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    if len(inputs) == 3:
        x, a, ids = inputs
    else:
        x, a = inputs
        ids = None

    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    kwargs = dict(
        kernel_regularizer=reg,
        bias_regularizer=reg,
    )

    x = dropout(x, input_dropout_rate)
    x = tf.keras.layers.Dense(
        hidden_size,
        **kwargs,
    )(x)
    if normalization:
        x = normalization(x)
    x = activation(x)
    x = dropout(x, dropout_rate)
    logits = tf.keras.layers.Dense(num_classes, **kwargs)(x)
    logits = Krylov(num_propagations)([a, logits])
    if simplified:
        logits = tf.reduce_sum(logits, axis=1)
    else:
        logits = GatedSum(**kwargs)(logits)
    if ids is not None:
        logits = tf.gather(logits, ids, axis=0)
    model = tf.keras.Model(inputs, logits)
    return model

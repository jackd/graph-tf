import tensorflow as tf

from graph_tf.projects.gat_v2 import layers as gat_layers
from graph_tf.utils.ops import sparse_gather


def gat(
    inputs_spec,
    num_classes: int,
    l2_reg: float = 2.5e-4,
    hidden_units: int = 8,
    hidden_heads: int = 8,
    final_heads: int = 1,
    dropout_rate: float = 0.6,
):
    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
    if len(inputs) == 3:
        x_in, a_in, ids = inputs
    else:
        x_in, a_in = inputs
        ids = None

    reg = tf.keras.regularizers.l2(l2_reg)

    x = gat_layers.GATConv(
        hidden_units,
        num_heads=hidden_heads,
        reduction=None,
        dropout_rate=dropout_rate,
        activation="elu",
        kernel_regularizer=reg,
        bias_regularizer=reg,
        attn_kernel_regularizer=reg,
        attn_bias_regularizer=reg,
        name="initial",
    )([x_in, a_in])
    x = tf.reshape(x, (-1, hidden_units * hidden_heads))
    if ids is not None:
        a_in = sparse_gather(a_in, ids).st
    x = gat_layers.GATConv(
        num_classes,
        num_heads=final_heads,
        reduction="mean",
        dropout_rate=dropout_rate,
        activation=None,
        kernel_regularizer=reg,
        attn_kernel_regularizer=reg,
        attn_bias_regularizer=reg,
        bias_regularizer=reg,
        name="final",
    )([x, a_in])
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

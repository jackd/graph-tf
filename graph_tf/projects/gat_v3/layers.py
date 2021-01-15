import tensorflow as tf

from graph_tf.projects.gat_v2 import ops  # not a typo
from graph_tf.utils.layers import MultiHeadedDense
from graph_tf.utils.ops import segment_softmax

activations = tf.keras.activations
constraints = tf.keras.constraints
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers


def rescale_initializer(init, factor=2):
    init = initializers.get(init)
    return initializers.VarianceScaling(
        scale=init.scale * factor,
        mode=init.mode,
        distribution=init.distribution,
        seed=init.seed,
    )


class GATConv(tf.keras.layers.Layer):
    """Similar to v2 but uses a dot attention mechanism."""

    def __init__(
        self,
        channels: int,
        dot_dim: int = 8,
        num_heads: int = 1,
        dropout_rate=0.5,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        attn_bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
        reduction=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.num_heads = num_heads
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.dot_dim = dot_dim
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.attn_bias_regularizer = regularizers.get(attn_bias_regularizer)
        self.activation = activations.get(activation)

        self.attn_dense = MultiHeadedDense(
            2 * dot_dim,
            name="attn-dense",
            kernel_regularizer=self.attn_kernel_regularizer,
            kernel_constraint=self.attn_kernel_constraint,
            # rescale by 2 because that's almost equivalent to two separate dense layers
            # kernel_initializer=rescale_initializer(self.attn_kernel_initializer, 2),
            kernel_initializer=self.attn_kernel_initializer,
            bias_regularizer=self.attn_bias_regularizer,
        )
        self.values_dense = MultiHeadedDense(
            channels,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            kernel_constraint=kernel_constraint,
            use_bias=False,
            name="values-dense",
        )
        self.input_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.values_dropout = tf.keras.layers.Dropout(dropout_rate)

        self.use_bias = use_bias
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self._bias_shape = (num_heads, channels) if reduction is None else (channels,)
        self.reduction = reduction
        self.bias = None

    def build(self, input_shape):
        if not self.built:
            x_shape, _ = input_shape
            N, F = x_shape
            H = self.num_heads
            x_shape = (N, H, F)
            self.values_dense.build(x_shape)
            x_shape = self.values_dense.compute_output_shape(x_shape)
            self.attn_dense.build(x_shape)
            if self.use_bias:
                self.bias = self.add_weight(
                    "bias",
                    constraint=self.bias_constraint,
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    shape=self._bias_shape,
                )
            return super().build(input_shape)

    def call(self, inputs, training=None):
        x, a = inputs
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.num_heads, 1))
        x = self.input_dropout(x, training=training)
        x = self.values_dense(x)
        q, k = tf.split(  # pylint: disable=redundant-keyword-arg,no-value-for-parameter
            self.attn_dense(x), 2, axis=-1
        )

        row, col = tf.unstack(a.indices, axis=-1)
        q = tf.gather(q, row, axis=0)
        k = tf.gather(k, col, axis=0)
        # attn = tf.keras.backend.dot(q, k)
        attn = tf.einsum("ehd,ehd->eh", q, k)
        attn = segment_softmax(attn, row)
        attn = self.attn_dropout(attn, training=training)

        values = self.values_dropout(x, training=training)
        out = ops.multi_attention_v0(values, attn, a)
        if self.reduction == "sum":
            out = tf.reduce_sum(out, axis=1)
        elif self.reduction == "mean":
            out = tf.reduce_mean(out, axis=1)
        else:
            assert self.reduction is None
        if self.use_bias:
            out = out + self.bias
        return self.activation(out)

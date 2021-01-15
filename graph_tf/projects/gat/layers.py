import tensorflow as tf

from graph_tf.utils.layers import SparseDropout

# from graph_tf.utils import ops

activations = tf.keras.activations
constraints = tf.keras.constraints
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers


class GATConvSingle(tf.keras.layers.Layer):
    """GAT implementation consistent with the paper author's implementation."""

    def __init__(
        self,
        channels,
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
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.attn_bias_regularizer = regularizers.get(attn_bias_regularizer)
        self.activation = activations.get(activation)

        kwargs = dict(
            kernel_regularizer=attn_kernel_regularizer,
            kernel_constraint=attn_kernel_constraint,
            kernel_initializer=attn_kernel_initializer,
            bias_regularizer=attn_bias_regularizer,
        )
        self.key_dense = tf.keras.layers.Dense(
            1, use_bias=False, name="key-dense", **kwargs
        )
        self.query_dense = tf.keras.layers.Dense(
            1, use_bias=False, name="query-dense", **kwargs
        )
        # self.attn_dense = tf.keras.layers.Dense(
        #     2, use_bias=False, name="key-dense", **kwargs
        # )
        self.values_dense = tf.keras.layers.Dense(
            channels,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            kernel_constraint=kernel_constraint,
            use_bias=False,
            name="values-dense",
        )
        self.input_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.attn_dropout = SparseDropout(dropout_rate)
        self.values_dropout = tf.keras.layers.Dropout(dropout_rate)

        self.use_bias = use_bias
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias = None

    def build(self, input_shape):
        if self.built:
            return
        x_shape, a_shape = input_shape
        del a_shape

        with tf.name_scope(self.name_scope()):  # pylint: disable=not-callable
            with tf.name_scope("value"):
                self.values_dense.build(x_shape)
            x_shape = self.values_dense.compute_output_shape(x_shape)
            # with tf.name_scope("attn"):
            #     self.attn_dense.build(x_shape)
            with tf.name_scope("query"):
                self.query_dense.build(x_shape)
            with tf.name_scope("key"):
                self.key_dense.build(x_shape)

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                shape=(self.channels,),
            )
        return super().build(input_shape)

    def call(self, inputs, training=None):
        x, a = inputs
        x = self.input_dropout(x, training=training)
        x = self.values_dense(x)

        # query, key = tf.unstack(self.attn_dense(x), axis=-1)
        query = tf.squeeze(self.query_dense(x), axis=-1)
        key = tf.squeeze(self.key_dense(x), axis=-1)

        indices = a.indices
        row, col = tf.unstack(indices, axis=-1)
        query = tf.gather(query, row, axis=0)
        key = tf.gather(key, col, axis=0)
        attn = tf.nn.leaky_relu(query + key)
        attn_st = tf.SparseTensor(indices, attn, a.dense_shape)
        attn_st = tf.sparse.softmax(attn_st)  # pylint: disable=no-value-for-parameter
        attn_st = self.attn_dropout(attn_st, training=training)

        values = self.values_dropout(x, training=training)
        values = tf.sparse.sparse_dense_matmul(attn_st, values)
        if self.use_bias:
            values = values + self.bias
        if self.activation is not None:
            values = self.activation(values)

        return values


class GATConv(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads=1, concat_heads=True, name=None, **kwargs):
        super().__init__(name=name)
        self._concat_heads = concat_heads
        self._heads = tuple(
            GATConvSingle(*args, **kwargs, name=f"head-{i}") for i in range(num_heads)
        )

    def build(self, input_shape):
        # with tf.name_scope(self.name_scope()):
        for head in self._heads:
            head.build(input_shape)
        return super().build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        outputs = [head(inputs, training=training) for head in self._heads]
        if self._concat_heads:
            outputs = tf.concat(  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
                outputs, axis=-1
            )
        else:
            outputs = tf.add_n(outputs) / len(outputs)
        return outputs

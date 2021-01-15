import tensorflow as tf

from graph_tf.projects.dfnet.ops import dfnet_conv

activations = tf.keras.activations
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers
constraints = tf.keras.constraints


class DFNetConv(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dim: int,
        num_filters=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.num_filters = num_filters

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_dim = None
        self.ar_kernel = None
        self.ma_kernel = None
        self.bias = None

    def build(self, input_shape):
        if self.built:
            return

        x_shape, signal_shape = input_shape[:2]

        self.input_dim = x_shape[-1]

        if self.num_filters is not None:
            ar_kernel_shape = (self.num_filters * self.input_dim, self.output_dim)
            ma_kernel_shape = (
                self.num_filters * signal_shape[1],
                self.output_dim,
            )
        else:
            ar_kernel_shape = (self.input_dim, self.output_dim)
            ma_kernel_shape = (signal_shape[1], self.output_dim)

        self.ar_kernel = self.add_weight(
            shape=ar_kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.ma_kernel = self.add_weight(
            shape=ma_kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.kernel_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        super().build(input_shape)

    def call(self, inputs):
        x, signal, arma_conv_AR, arma_conv_MA = inputs
        output = dfnet_conv(
            x,
            self.num_filters,
            arma_conv_AR,
            arma_conv_MA,
            signal,
            self.ar_kernel,
            self.ma_kernel,
        )
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_filters": self.num_filters,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config

from typing import Optional, Tuple

import tensorflow as tf

from graph_tf.utils import ops


class SparseDense(tf.keras.layers.Dense):
    """Dense implementation for `SparseTensor` inputs."""

    def call(self, inputs: tf.SparseTensor):
        return ops.sparse_dense(
            inputs, kernel=self.kernel, bias=self.bias, activation=self.activation
        )


class SparseDropout(tf.keras.layers.Layer):
    """Dropout implementation for `SparseTensor` inputs."""

    def __init__(self, rate: float, seed: Optional[int] = None, **kwargs):
        self.rate = rate
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(dict(rate=self.rate, seed=self.seed))
        return config

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        if training is None:
            training = tf.keras.backend.learning_phase()

        def train_fn():
            return ops.sparse_dropout(inputs, rate=self.rate, seed=self.seed)

        def val_fn():
            return inputs

        return ops.smart_cond(training, train_fn, val_fn)


class ConvolutionBase(tf.keras.layers.Layer):
    """Base class for convolution layers that use a kernel and bias."""

    def __init__(
        self,
        filters: int,
        use_bias: bool = True,
        activation=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs
    ):
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.filters = filters

        self.kernel = None
        self.bias = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                kernel_initializer=tf.keras.utils.serialize_keras_object(
                    self.kernel_initializer
                ),
                kernel_regularizer=tf.keras.utils.serialize_keras_object(
                    self.kernel_regularizer
                ),
                kernel_constraint=tf.keras.utils.serialize_keras_object(
                    self.kernel_constraint
                ),
                bias_initializer=tf.keras.utils.serialize_keras_object(
                    self.bias_initializer
                ),
                bias_regularizer=tf.keras.utils.serialize_keras_object(
                    self.bias_regularizer
                ),
                bias_constraint=tf.keras.utils.serialize_keras_object(
                    self.bias_constraint
                ),
                use_bias=self.use_bias,
                filters=self.filters,
                activation=tf.keras.utils.serialize_keras_object(self.activation),
            )
        )
        return config

    def _kernel_shape(self, input_shape) -> Tuple[int, ...]:
        return (input_shape[-1], self.filters)

    def _bias_shape(self, input_shape) -> Tuple[int, ...]:
        del input_shape
        return (self.filters,)

    def build(self, input_shape):
        if not self.built:
            self.kernel = self.add_weight(
                name="kernel",
                shape=self._kernel_shape(input_shape),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
            if self.use_bias:
                self.bias = self.add_weight(
                    name="bias",
                    shape=self._bias_shape(input_shape),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )
        return super().build(input_shape)

    def _finalize(self, x: tf.Tensor):
        return ops.finalize_dense(x, self.bias, self.activation)


class MultiHeadedDense(ConvolutionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = None
        self.filters_in = None

    def _kernel_shape(self, input_shape):
        return (self.num_heads, self.filters_in, self.filters)

    def _bias_shape(self, input_shape):
        return (self.num_heads, self.filters)

    def build(self, input_shape):
        if not self.built:
            assert len(input_shape) == 3
            self.num_heads, self.filters_in = input_shape[1:]
            self.kernel = self.add_weight(
                name="kernel",
                shape=self._kernel_shape(input_shape),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
            if self.use_bias:
                self.bias = self.add_weight(
                    name="bias",
                    shape=self._bias_shape(input_shape),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )
        return super().build(input_shape)

    def call(self, inputs):
        x = tf.einsum("nhj,hji->nhi", inputs, self.kernel)
        return self._finalize(x)


class Krylov(tf.keras.layers.Layer):
    def __init__(self, dims: int, axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.dims = dims
        self.axis = axis

    def get_config(self):
        config = super().get_config()
        config["dims"] = self.dims
        return config

    def call(self, inputs):
        A, b = inputs
        return ops.krylov(A, b, dims=self.dims, axis=self.axis)

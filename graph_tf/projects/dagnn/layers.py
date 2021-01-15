import tensorflow as tf


class GatedSum(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gate = tf.keras.layers.Dense(
            1,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            activation="sigmoid",
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            dict(
                kernel_initializer=tf.keras.utils.serialize_keras_object(
                    self.gate.kernel_initializer
                ),
                kernel_regularizer=tf.keras.utils.serialize_keras_object(
                    self.gate.kernel_regularizer
                ),
                bias_initializer=tf.keras.utils.serialize_keras_object(
                    self.gate.bias_initializer
                ),
                bias_regularizer=tf.keras.utils.serialize_keras_object(
                    self.gate.bias_regularizer
                ),
            )
        )
        return config

    def call(self, inputs):
        if isinstance(inputs, tf.Tensor):
            gate_features = unscaled_features = inputs
        else:
            gate_features, unscaled_features = inputs
        scale = tf.squeeze(self.gate(gate_features), axis=-1)
        return tf.linalg.matvec(unscaled_features, scale, transpose_a=True)

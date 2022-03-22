import typing as tp

import tensorflow as tf

from graph_tf.utils import layers as _base_layers
from graph_tf.utils.type_checks import is_sparse_tensor


@tf.keras.utils.register_keras_serializable(package="fgcn")
class FactorizedGraphConvolution(_base_layers.ConvolutionBase):
    def __init__(self, *args, transform_first: tp.Optional[bool] = None, **kwargs):
        self.transform_first = transform_first
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config["transform_first"] = self.transform_first

    def _kernel_shape(self, input_shape) -> tp.Tuple[int, ...]:
        return (input_shape[0][-1], self.filters)

    def call(self, inputs):
        def transform(x):
            if is_sparse_tensor(x):
                return tf.sparse.sparse_dense_matmul(x, self.kernel)
            return tf.matmul(x, self.kernel)

        features, V = inputs
        features.shape.assert_has_rank(2)
        V.shape.assert_has_rank(2)
        tf.debugging.assert_equal(tf.shape(V)[0], tf.shape(features)[0])

        transform_first = self.transform_first
        if transform_first is None:
            transform_first = features.shape[1] > self.filters

        if transform_first:
            features = transform(features)

        features = tf.matmul(V, tf.matmul(V, features, transpose_a=True))
        if not transform_first:
            features = transform(features)

        return self._finalize(features)

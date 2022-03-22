import typing as tp

import tensorflow as tf

from graph_tf.projects.gcn import ops


class PropagationInput(tp.NamedTuple):
    A: tp.Union[tf.Tensor, tf.SparseTensor]
    x: tf.Tensor


class Propagation(tf.keras.layers.Layer):  # pylint: disable=too-few-public-methods
    def call(self, inputs: PropagationInput):  # pylint: disable=no-self-use
        A, x = inputs
        return ops.matmul(A, x)


# class MultiPropagation(Propagation):
#     def __init__(self, num_propagations: int, **kwargs):
#         super().__init__(**kwargs)
#         self.num_propagations = num_propagations

#     def get_config(self):
#         config = super().get_config()
#         config["num_propagations"] = self.num_propagations

#     def call(self, inputs: Propagation):
#         A, x = inputs
#         terms = [x]
#         for _ in range(self.num_propagations):
#             x = ops.matmul(A, x)
#             terms.append(x)
#         return tf.add_n(terms)


class StalePropagationInput(tp.NamedTuple):
    A: tp.Union[tf.Tensor, tf.SparseTensor]
    x: tf.Tensor
    x0: tf.Tensor
    y0: tf.Tensor


class StalePropagation(tf.keras.layers.Layer):
    def __init__(self, fresh_layer: Propagation, **kwargs):
        super().__init__(**kwargs)
        self._x0 = None
        self._y0 = None
        if not isinstance(fresh_layer, tf.keras.layers.Layer):
            fresh_layer = tf.keras.utils.deserialize_keras_object(fresh_layer)
        self.fresh_layer = fresh_layer

    def get_config(self):
        config = super().get_config()
        config["fresh_layer"] = tf.keras.utils.serialize_keras_object(self.fresh_layer)
        return config

    def build_and_call(self, A: tp.Union[tf.Tensor, tf.SparseTensor], x: tf.Tensor):
        x.shape.assert_has_rank(2)
        A.shape.assert_has_rank(2)
        self.build_inputs(x.shape[1])
        return self(StalePropagationInput(A, x, self.x0, self.y0))

    def build_inputs(self, size: int):
        if self._x0 is None:
            self._x0 = tf.keras.Input((size,), dtype=self.dtype, name=f"{self.name}_x0")
            self._y0 = tf.keras.Input((size,), dtype=self.dtype, name=f"{self.name}_y0")
        else:
            assert self._x0.shape[1] == size, (self._x0.shape, size)
            assert self._y0.shape[1] == size, (self._y0.shape, size)

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    def call(self, inputs):
        A, x, x0, y0 = inputs
        if x0 is not None:
            x = x - x0

        out = self.fresh_layer(PropagationInput(A, x))
        if y0 is not None:
            out = out + y0
        return out

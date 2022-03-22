import typing as tp

import tensorflow as tf

from graph_tf.utils import ops


class PropagationInput(tp.NamedTuple):
    T: tp.Union[tf.Tensor, tf.SparseTensor]  # transition matrix
    X: tf.Tensor  # node features


class Propagation(tf.keras.layers.Layer):
    def call(self, inp: PropagationInput):
        T, X = inp
        out = ops.matmul(T, X)
        out.set_shape(X.shape)
        return out

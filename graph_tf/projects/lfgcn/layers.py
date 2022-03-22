import typing as tp

import tensorflow as tf

from graph_tf.utils import layers as _base_layers
from graph_tf.utils.ops import matmul


class LearnedFactorizedGraphConvolution(_base_layers.ConvolutionBase):
    """
    Perform Learned Factorized Graph Convolution.
    """

    def __init__(self, *args, reg_coeff: float = 1e-3, **kwargs):
        self.reg_coeff = reg_coeff
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(reg_coeff=self.reg_coeff)
        return config

    def _kernel_shape(self, input_shape):
        return (input_shape[0][-1], self.filters)

    def call(
        self,
        inputs: tp.Tuple[
            tp.Union[tf.Tensor, tf.SparseTensor], tf.Tensor, tf.SparseTensor
        ],
    ):
        x, V, adjacency = inputs

        x = matmul(x, self.kernel)

        x_ = matmul(V, matmul(V, x, transpose_a=True))

        tf.stop_gradient(x)
        self.add_loss(
            self.reg_coeff
            * tf.reduce_sum(
                tf.math.squared_difference(
                    # matmul(adjacency, xs),
                    # matmul(V, matmul(V, xs, transpose_a=True))
                    matmul(adjacency, x),
                    x_,
                )
            )
        )
        x = x_

        return self._finalize(x)

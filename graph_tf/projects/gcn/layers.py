from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf

from graph_tf.projects.gcn import ops
from graph_tf.utils import layers as _base_layers
from graph_tf.utils.ops import SparseImplementation


class GraphConvolution(_base_layers.ConvolutionBase):
    """
    Perform graph convolution with a single adjacency matrix.

    This differs from the original implementation in 2 ways:
    1. It does not include dropout.
    2. It does not support multiple adjacency matrices.

    To resolve (1), see
    - `graph_tf.utils.layers.SparseDropout`; or
    - `tf.keras.layers.Dropout`.

    To resolve (2), see MultiGraphConvolution.
    """

    def __init__(
        self,
        *args,
        sparse_impl: str = SparseImplementation.COO,
        transform_first: Optional[bool] = None,
        **kwargs
    ):
        self.sparse_impl = sparse_impl
        self.transform_first = transform_first
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config["sparse_impl"] = self.sparse_impl
        config["transform_first"] = self.transform_first
        return config

    def _kernel_shape(self, input_shape):
        return (input_shape[0][-1], self.filters)

    def call(
        self, inputs: Tuple[Union[tf.Tensor, tf.SparseTensor], tf.SparseTensor],
    ):
        x, adjacency = inputs
        x = ops.graph_conv(
            x,
            adjacency,
            self.kernel,
            self.sparse_impl,
            transform_first=self.transform_first,
        )
        return self._finalize(x)


class MultiGraphConvolution(GraphConvolution):
    """GraphConvolution with multiple adjacency matrices."""

    def _kernel_shape(self, input_shape):
        return (input_shape[0][-1], len(input_shape[1]), self.filters)

    def call(
        self,
        inputs: Tuple[Union[tf.Tensor, tf.SparseTensor], Sequence[tf.SparseTensor]],
    ):
        x, adjacencies = inputs
        x = ops.multi_graph_conv_v0(
            x,
            adjacencies,
            self.kernel,
            self.sparse_impl,
            transform_first=self.transform_first,
        )
        return self._finalize(x)

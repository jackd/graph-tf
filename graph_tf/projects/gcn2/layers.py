import typing as tp

import tensorflow as tf

from graph_tf.projects.gcn2.ops import graph_conv
from graph_tf.utils import layers as _base_layers


class GraphConvolutionInput(tp.NamedTuple):
    adjacency: tp.Union[tf.Tensor, tf.SparseTensor]
    features: tf.Tensor
    features0: tf.Tensor


register_keras_serializable = tf.keras.utils.register_keras_serializable(
    package="graph_tf.gcn2"
)


@register_keras_serializable
class GraphConvolution(_base_layers.ConvolutionBase):
    def __init__(
        self,
        filters: int,
        beta: float,
        alpha: float,
        use_bias: bool = True,
        kernel_initializer=None,
        bias_initializer="zeros",
        variant: bool = False,
        **kwargs,
    ):

        if kernel_initializer is None:
            kernel_initializer = tf.keras.initializers.VarianceScaling(
                scale=1 / 3, mode="fan_out", distribution="uniform"
            )
        self.beta = beta
        self.alpha = alpha
        self.variant = variant
        super().__init__(
            filters=filters,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=use_bias,
            **kwargs,
        )

    def get_config(self) -> tp.Dict[str, tp.Any]:
        config: dict = super().get_config()
        config.update(alpha=self.alpha, beta=self.beta, variant=self.variant)
        return config

    def _kernel_shape(self, input_shape) -> tp.Tuple[int, ...]:
        return (input_shape[-1][-1], self.filters)

    def build(self, input_shape):
        if self.built:
            return
        super().build(input_shape)

    def call(self, inputs: GraphConvolutionInput, training=False):
        adjacency, features, features0 = inputs
        x = graph_conv(
            adjacency,
            features,
            features0,
            self.kernel,
            alpha=self.alpha,
            beta=self.beta,
            variant=self.variant,
        )
        return self._finalize(x)

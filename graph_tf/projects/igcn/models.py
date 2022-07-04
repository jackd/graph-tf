import functools
import typing as tp

import gin
import tensorflow as tf

register = functools.partial(gin.register, module="gtf.igcn.models")


class Matmul(tf.keras.layers.Layer):
    def call(self, args):
        a, b = args
        return a @ b


@register
def logit_propagated_model(
    input_spec: tp.Tuple[tf.TypeSpec, tf.TensorSpec],
    num_classes: int,
    mlp_fn: tp.Callable[[tp.Any, int], tf.keras.Model],
    branched: bool = False,
):
    prop_spec, x_spec = input_spec
    # hack to get around keras TypeSpec assertions
    if prop_spec.shape[1] is not None:
        prop_spec = tf.TensorSpec(
            shape=(*prop_spec.shape[:-1], None), dtype=prop_spec.dtype
        )
    mlp = mlp_fn(x_spec, num_classes * 2 if branched else num_classes)
    x = mlp.inputs
    prop = tf.keras.Input(type_spec=prop_spec)
    if branched:
        (skip, to_smooth) = tf.split(
            mlp.output, 2, axis=-1
        )  # pylint: disable=redundant-keyword-arg,no-value-for-parameter
        smoothed = Matmul()((prop, to_smooth))
        logits = skip + smoothed
    else:
        logits = Matmul()((prop, mlp.output))
    return tf.keras.Model((prop, x), logits)

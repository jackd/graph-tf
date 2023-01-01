import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.utils import models as model_utils

register = functools.partial(gin.register, module="gtf.igcn.models")


class AddBias(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = None

    def build(self, input_shape):
        if self.built:
            return
        self.bias = self.add_weight(
            "bias", shape=input_shape[-1:], initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.bias


@register
def logit_propagated_model(
    input_spec: tp.Tuple[
        tp.Union[tf.TypeSpec, tp.Sequence[tf.TypeSpec]], tf.TensorSpec
    ],
    num_classes: int,
    mlp_fn: tp.Callable[[tp.Any, int], tf.keras.Model],
    add_final_bias: bool = False,
):
    prop_spec, x_spec = input_spec

    # hack to get around keras TypeSpec assertions
    def fix_prop_spec(spec):
        if spec.shape[0] is not None:
            spec = tf.TensorSpec(shape=(None, *spec.shape[1:]), dtype=spec.dtype)
        return spec

    props = tf.nest.map_structure(
        lambda spec: tf.keras.Input(type_spec=fix_prop_spec(spec)), prop_spec
    )
    # props = tf.nest.map_structure(
    #     lambda spec: tf.keras.Input(type_spec=spec), prop_spec
    # )
    single_prop = tf.is_tensor(props)
    mlp = mlp_fn(x_spec, num_classes if single_prop else num_classes * len(props))
    x = mlp.inputs
    if single_prop:
        logits = props @ mlp.output
    else:
        logs = tf.split(mlp.output, len(props), axis=1)
        logits = tf.add_n([prop @ log for prop, log in zip(props, logs)])
    if add_final_bias:
        logits = AddBias()(logits)
    return tf.keras.Model((props, x), logits)


class Propagator(tf.keras.layers.Layer):
    def call(self, inputs):
        prop, x = inputs
        tf.debugging.assert_all_finite(x, "propagation inputs finite")
        x = prop @ x
        tf.debugging.assert_all_finite(x, "propagation outputs finite")
        return x


@register
def logit_propagated_model_v2(
    input_spec: tp.Tuple[
        tp.Union[tf.TypeSpec, tp.Sequence[tf.TypeSpec]], tf.TensorSpec
    ],
    num_classes: int,
    hidden_units: tp.Iterable[int] = (64,),
    dropout_rate: float = 0.0,
    dense_fn: tp.Callable[[tf.Tensor, int], tf.Tensor] = model_utils.dense,
    activation="relu",
):
    prop_spec, x_spec = input_spec

    # hack to get around keras TypeSpec assertions
    def fix_prop_spec(spec):
        if spec.shape[0] is not None:
            spec = tf.TensorSpec(shape=(None, *spec.shape[1:]), dtype=spec.dtype)
        return spec

    props = tf.nest.map_structure(
        lambda spec: tf.keras.Input(type_spec=fix_prop_spec(spec)), prop_spec
    )
    assert tf.is_tensor(props)

    activation = tf.keras.activations.get(activation)

    # if input_dropout_rate is None:
    #     input_dropout_rate = dropout_rate

    x = tf.keras.Input(type_spec=x_spec)
    inputs = (props, x)
    # x = model_utils.dropout(x, input_dropout_rate)
    for u in hidden_units:
        x = model_utils.dropout(x, dropout_rate)
        x = dense_fn(x, u)
        x = activation(x)
    x = model_utils.dropout(x, dropout_rate)
    # x = props @ x
    x = Propagator()((props, x))
    x = model_utils.dropout(x, dropout_rate)
    logits = dense_fn(x, num_classes)
    return tf.keras.Model(inputs, logits)

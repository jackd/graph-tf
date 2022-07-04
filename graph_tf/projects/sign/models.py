import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.data.single import DataSplit, SemiSupervisedSingle
from graph_tf.data.transforms import transformed
from graph_tf.utils.models import dense, mlp

register = functools.partial(gin.register, module="gtf.sign")


@register
def preprocess_single(
    data: SemiSupervisedSingle,
    features_transform: tp.Iterable[tp.Callable] = (),
    adjacency_transform: tp.Iterable[tp.Callable] = (),
    features_fn: tp.Iterable[tp.Callable[[tf.SparseTensor, tf.Tensor], tf.Tensor]] = (),
    identity_features: bool = True,
    batch_size: tp.Optional[int] = None,
    # reorthogonalize: bool = False,
) -> DataSplit:
    f0 = transformed(data.node_features, features_transform)
    transition = transformed(data.adjacency, adjacency_transform)
    node_features = [f0] if identity_features else []
    node_features.extend([fn(transition, f0) for fn in features_fn])

    def get_data(ids, training=False):
        if ids is None:
            return None
        tensors = (
            tuple(tf.gather(f, ids, axis=0) for f in node_features),
            tf.gather(data.labels, ids, axis=0),
        )
        if batch_size is None:
            dataset = tf.data.Dataset.from_tensors(tensors)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(tensors)
            if training:
                dataset = dataset.shuffle(ids.shape[0])
            dataset = dataset.batch(batch_size)
        return dataset

    return DataSplit(
        get_data(data.train_ids, training=True),
        get_data(data.validation_ids),
        get_data(data.test_ids),
    )


@register
def sign(
    inputs_spec,
    num_classes: int,
    hidden_units: int = 256,
    dropout_rate: float = 0.0,
    input_dropout_rate: tp.Optional[float] = None,
    normalization: tp.Optional[tp.Callable[[tf.Tensor], tf.Tensor]] = None,
    activation="relu",
    final_activation=None,
    dense_fn: tp.Callable[[tf.Tensor], tf.Tensor] = dense,
):
    """
    Based on the implementation at

    https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/sign/sign.py.
    """
    activation = tf.keras.activations.get(activation)
    kwargs = dict(
        dense_fn=dense_fn,
        activation=activation,
        dropout_rate=dropout_rate,
        input_dropout_rate=input_dropout_rate,
        normalization=normalization,
        hidden_units=hidden_units,
    )

    inputs = tuple(tf.keras.Input(type_spec=s) for s in inputs_spec)

    outputs = [mlp(inp, output_units=hidden_units, **kwargs).output for inp in inputs]
    x = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        outputs, axis=1
    )
    if normalization:
        x = normalization(x)
    x = activation(x)
    output = mlp(
        x, output_units=num_classes, final_activation=final_activation, **kwargs
    ).output
    return tf.keras.Model(inputs, output)

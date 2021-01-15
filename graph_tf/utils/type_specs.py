from typing import Union

import tensorflow as tf


def keras_input(
    spec: Union[
        tf.TensorSpec,
        tf.SparseTensorSpec,
        tf.RaggedTensorSpec,
        tf.Tensor,
        tf.RaggedTensor,
        tf.SparseTensor,
    ],
    use_batch_size: bool = False,
):
    kwargs = dict(shape=spec.shape[1:], dtype=spec.dtype)
    if use_batch_size:
        kwargs["batch_size"] = spec.shape[0]
    if isinstance(spec, (tf.SparseTensorSpec, tf.SparseTensor)):
        kwargs["sparse"] = True
    elif isinstance(spec, (tf.RaggedTensorSpec, tf.RaggedTensor)):
        kwargs["ragged"] = True
    else:
        assert isinstance(spec, (tf.TensorSpec, tf.Tensor))
    return tf.keras.Input(**kwargs)


def get_type_spec(tensors):
    return tf.nest.map_structure(tf.type_spec_from_value, tensors)

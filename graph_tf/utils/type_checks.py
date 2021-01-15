import tensorflow as tf


def is_dense_tensor(x) -> bool:
    return (
        isinstance(x, tf.Tensor)
        or tf.is_tensor(x)
        and tf.keras.backend.is_keras_tensor(x)
        and isinstance(x.type_spec, tf.TensorSpec)
    )


def is_sparse_tensor(x) -> bool:
    return (
        isinstance(x, tf.SparseTensor)
        or tf.is_tensor(x)
        and tf.keras.backend.is_keras_tensor(x)
        and isinstance(x.type_spec, tf.SparseTensorSpec)
    )


def is_ragged_tensor(x) -> bool:
    return (
        isinstance(x, tf.RaggedTensor)
        or tf.is_tensor(x)
        and tf.keras.backend.is_keras_tensor(x)
        and isinstance(x.type_spec, tf.RaggedTensorSpec)
    )

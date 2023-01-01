import functools
import typing as tp

import gin
import tensorflow as tf
import tqdm

from graph_tf.data.single import (
    DataSplit,
    SemiSupervisedSingle,
    SparseTensorTransform,
    TensorTransform,
    get_largest_component,
    preprocess_weights,
    transformed,
)
from graph_tf.utils.linalg import eigsh

register = functools.partial(gin.register, module="gtf.fgcn.data")


@register
def get_eigen_factorization(A: tf.SparseTensor, rank: int = 128):
    w, v = eigsh(A, rank, which="LM")  # pylint: disable=unpacking-non-sequence
    V = v * tf.sqrt(w)
    return V


@register
def get_learned_factorization(
    A: tf.SparseTensor,
    rank: int = 128,
    optimizer: tp.Optional[tf.keras.optimizers.Optimizer] = None,
    steps: int = 1000,
) -> tf.Tensor:
    learning_rate = 1.0e-1
    V = get_eigen_factorization(A, rank)
    V = tf.Variable(V, trainable=True, dtype=tf.float64)
    if optimizer is None:
        # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    A = tf.sparse.to_dense(A)
    A = tf.cast(A, tf.float64)

    @tf.function
    def train_step(A):
        with tf.GradientTape() as tape:
            tape.watch(V)
            vvt = tf.matmul(V, V, transpose_b=True)
            loss = tf.reduce_sum(tf.math.squared_difference(vvt, A))
            var_list = [V]
            grads = tape.gradient(loss, var_list)

        # tf.print(tf.reduce_max(tf.abs(grads[0])))
        # V = V - learning_rate * grads[0]
        optimizer.apply_gradients(zip(grads, var_list))
        return loss

    prog = tqdm.trange(steps, desc="Training factorisation...")
    for _ in prog:
        loss = train_step(A)
        # print(loss.numpy())
        prog.desc = f"Training factorisation. Loss = {loss.numpy():.10f}"
    return tf.convert_to_tensor(V.numpy(), dtype=tf.float32)


@register
def get_prefactorized_data(
    data: SemiSupervisedSingle,
    adjacency_transform: tp.Union[
        SparseTensorTransform, tp.Sequence[SparseTensorTransform]
    ] = (),
    features_transform: tp.Union[TensorTransform, tp.Sequence[TensorTransform]] = (),
    factoriser: tp.Callable = get_eigen_factorization,
    normalize: bool = False,
    largest_component_only: bool = False,
) -> DataSplit:
    if largest_component_only:
        data = get_largest_component(data, directed=False)
    features = transformed(data.node_features, features_transform)
    A = transformed(data.adjacency, adjacency_transform)
    n = A.shape[0]
    assert n is not None

    V = factoriser(A)
    if normalize:
        d = tf.matmul(
            V, tf.linalg.matvec(V, tf.ones((n,), dtype=V.dtype), transpose_a=True)
        )
        V = V * tf.math.rsqrt(tf.abs(d))
    inputs = (features, V)

    def get_data(ids):
        if ids is None:
            return None
        example = inputs, data.labels, preprocess_weights(ids, n, normalize=True)
        return tf.data.Dataset.from_tensors(example)

    return DataSplit(
        get_data(data.train_ids), get_data(data.validation_ids), get_data(data.test_ids)
    )

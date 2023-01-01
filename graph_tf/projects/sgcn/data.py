import typing as tp

import gin
import tensorflow as tf

from graph_tf.data.single import (
    DataSplit,
    SemiSupervisedSingle,
    TensorTransform,
    get_largest_component,
    preprocess_weights,
    transformed,
)
from graph_tf.data.transforms import laplacian, normalized_laplacian
from graph_tf.utils.arnoldi import ritz_embedding
from graph_tf.utils.linalg import eigsh_lap


@gin.configurable(module="gtf.sgcn.data")
def add(x, y):
    return x + y


@gin.configurable(module="gtf.sgcn.data")
def reciprocal(x):
    return 1 / x


@gin.configurable(module="gtf.sgcn.data")
def preprocess_single(
    data: SemiSupervisedSingle,
    *,
    num_eigs: int = 6,
    features_transform: tp.Union[TensorTransform, tp.Iterable[TensorTransform]] = (),
    eigenvalue_transform: tp.Union[TensorTransform, tp.Iterable[TensorTransform]] = (),
    largest_component_only: bool = False,
    normalized: bool = True,
    use_ritz_vectors: bool = False,
) -> DataSplit:
    if largest_component_only:
        data = get_largest_component(data, directed=False)
    num_nodes = data.adjacency.shape[0]
    L = (normalized_laplacian if normalized else laplacian)(data.adjacency)
    if use_ritz_vectors:
        w, v = ritz_embedding(L, tf.ones((num_nodes,)), num_eigs)
    else:
        w, v = eigsh_lap(L, k=num_eigs)  # pylint: disable=unpacking-non-sequence
    w = transformed(w, eigenvalue_transform)

    features = transformed(data.node_features, features_transform)
    inputs = features, w, v

    def get_split(ids: tf.Tensor):
        weights = preprocess_weights(ids, num_nodes, normalize=True)
        example = (inputs, data.labels, weights)
        return tf.data.Dataset.from_tensors(example)

    return DataSplit(
        get_split(data.train_ids),
        get_split(data.validation_ids),
        get_split(data.test_ids),
    )

from dataclasses import dataclass
from typing import Callable, Union

import gin
import networkx as nx
import tensorflow as tf
import tensorflow_datasets as tfds

from graph_tf.utils.ops import (
    indices_to_mask,
    renormalize_sparse,
    scatter_1d,
    sparse_boolean_mask,
    to_laplacian,
)
from graph_tfds.graphs import cite_seer, cora, pub_med  # pylint: disable=unused-import


@dataclass
class SemiSupervisedSingle:
    """Data class for a single sparsely labelled graph."""

    node_features: Union[tf.Tensor, tf.SparseTensor]  # [N, F]
    edges: tf.Tensor  # [E, 2]
    labels: tf.Tensor  # [N]
    train_ids: tf.Tensor  # [n_train << N]
    validation_ids: tf.Tensor  # [n_eval << N]
    test_ids: tf.Tensor  # [n_test < N]

    def __post_init__(self):
        self.node_features.shape.assert_has_rank(2)
        assert self.node_features.dtype.is_floating
        self.edges.shape.assert_has_rank(2)
        assert self.edges.shape[1] == 2
        assert self.edges.dtype.is_integer

        for ids in (self.train_ids, self.validation_ids, self.test_ids):
            ids.shape.assert_has_rank(1)
            assert ids.dtype.is_integer


def subgraph(single: SemiSupervisedSingle, indices: tf.Tensor):
    indices = tf.sort(tf.convert_to_tensor(indices, tf.int64))
    num_in = tf.shape(single.node_features, out_type=tf.int64)[0]
    num_out = tf.size(indices, out_type=tf.int64)
    mask = indices_to_mask(indices, num_in)
    remap = scatter_1d(indices, tf.range(num_out), num_in)

    edge_mask = tf.reduce_all(tf.gather(mask, single.edges, axis=0), axis=-1)

    node_features = single.node_features
    if isinstance(node_features, tf.SparseTensor):
        node_features = sparse_boolean_mask(node_features, mask, axis=0).st
    else:
        node_features = tf.boolean_mask(node_features, mask)
    edges = tf.gather(remap, tf.boolean_mask(single.edges, edge_mask), axis=0)
    labels = tf.boolean_mask(single.labels, mask)

    def valid_ids(ids):
        id_mask = tf.gather(mask, ids, axis=0)
        return tf.gather(remap, tf.boolean_mask(ids, id_mask), axis=0)

    return SemiSupervisedSingle(
        node_features,
        edges,
        labels,
        train_ids=valid_ids(single.train_ids),
        validation_ids=valid_ids(single.validation_ids),
        test_ids=valid_ids(single.test_ids),
    )


def get_largest_component(single: SemiSupervisedSingle):
    # create nx graph
    g = nx.Graph()
    for u, v in single.edges.numpy():
        g.add_edge(u, v)
    if nx.is_connected(g):
        return single

    _, _, component = max(
        ((len(c), i, c) for i, c in enumerate(nx.connected_components(g)))
    )
    return subgraph(single, tuple(component))


@gin.configurable(module="gtf.data")
def preprocess_laplacian(
    edges: tf.Tensor,
    num_nodes: Union[tf.Tensor, int],
    normalize: bool = True,
    dtype: tf.DType = tf.float32,
):
    num_edges = tf.shape(edges)[0]
    weights = tf.ones((num_edges,), dtype=dtype)
    adjacency = tf.SparseTensor(edges, weights, (num_nodes, num_nodes))
    return to_laplacian(adjacency, normalize=normalize)


@gin.configurable(module="gtf.data")
def preprocess_adjacency(
    edges: tf.Tensor,
    num_nodes: Union[tf.Tensor, int],
    add_self_loops: bool = True,
    normalize: bool = True,
    normalize_symmetric: bool = True,
    dtype: tf.DType = tf.float32,
):
    if isinstance(num_nodes, tf.Tensor):
        num_nodes.shape.assert_has_rank(0)
        assert num_nodes.dtype.is_integer
    num_edges = tf.shape(edges)[0]
    adj = tf.SparseTensor(
        indices=edges,
        values=tf.ones((num_edges,), dtype=dtype),
        dense_shape=(num_nodes, num_nodes),
    )

    if add_self_loops:
        adj = tf.sparse.add(adj, tf.sparse.eye(num_nodes, dtype=dtype))
    if normalize:
        adj = renormalize_sparse(adj, symmetric=normalize_symmetric)
    return adj


@gin.configurable(module="gtf.data")
def preprocess_weights(ids: tf.Tensor, num_nodes, normalize: bool = True):
    weights = indices_to_mask(ids, num_nodes, dtype=tf.float32)
    if normalize:
        weights = weights / tf.size(ids, out_type=tf.float32)
    return weights


@gin.configurable(module="gtf.data")
def preprocess_node_features(
    node_features: Union[tf.Tensor, tf.SparseTensor],
    as_dense: bool = True,
    normalize: bool = True,
):
    if isinstance(node_features, tf.SparseTensor) and as_dense:
        node_features = tf.sparse.to_dense(node_features)
    if normalize:
        if isinstance(node_features, tf.SparseTensor):
            row = node_features.indices[:, 0]
            factor = tf.math.segment_sum(node_features.values, row)
            node_features = node_features.with_values(
                node_features.values / tf.gather(factor, row, axis=0)
            )
        else:
            factor = tf.math.reduce_sum(node_features, axis=1, keepdims=True)
            factor = tf.where(factor == 0, tf.ones_like(factor), factor)
            node_features = node_features / factor
    return node_features


@gin.configurable(module="gtf.data")
def preprocess_single(
    data: SemiSupervisedSingle,
    features_fn: Callable[[tf.Tensor], tf.Tensor] = preprocess_node_features,
    adjacency_fn: Callable[
        [tf.Tensor, tf.Tensor], tf.SparseTensor
    ] = preprocess_adjacency,
    weighted: bool = False,
):
    node_features = data.node_features
    num_nodes = tf.shape(node_features, out_type=tf.int64)[0]
    node_features = features_fn(node_features)
    adjacency = adjacency_fn(data.edges, num_nodes)
    if weighted:

        def get_data(indices):
            return (
                (node_features, adjacency),
                data.labels,
                preprocess_weights(indices, num_nodes),
            )

    else:

        def get_data(indices):
            return (
                (node_features, adjacency, indices),
                tf.gather(data.labels, indices, axis=0),
            )

    return tuple(
        get_data(ids) for ids in (data.train_ids, data.validation_ids, data.test_ids)
    )


@gin.configurable(module="gtf.data")
def citations_data(
    name: str = "cora", largest_component_only: bool = False
) -> SemiSupervisedSingle:
    """
    Get semi-supervised citations data.

    Args:
        name: one of "cora", "cite_seer", "pub_med", or a registered tfds builder name
            with the same element spec.
        largest_component_only: if True, returns the subgraph associated with the
            largest connected component.
    """
    dataset = tfds.load(name)
    if isinstance(dataset, dict):
        if len(dataset) == 1:
            (dataset,) = dataset.values()
        else:
            raise ValueError(
                f"tfds builder {name} had more than 1 split ({sorted(dataset.keys())})."
                " Please use 'name/split'"
            )
    element = tf.data.experimental.get_single_element(dataset)
    graph = element["graph"]
    links = graph["links"]
    features = graph["node_features"]
    labels = element["node_labels"]
    train_ids = element["train_ids"]
    validation_ids = element["validation_ids"]
    test_ids = element["test_ids"]

    out = SemiSupervisedSingle(
        features, links, labels, train_ids, validation_ids, test_ids
    )
    if largest_component_only:
        out = get_largest_component(out)
    return out

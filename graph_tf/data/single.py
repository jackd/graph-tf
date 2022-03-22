import functools
import os
import typing as tp
from dataclasses import dataclass

import gin
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

import graph_tfds.graphs  # pylint: disable=unused-import
import stfu.ops
import tensorflow_datasets as tfds
from graph_tf.data.types import DataSplit
from graph_tf.utils.graph_utils import get_largest_component_indices
from graph_tf.utils.ops import indices_to_mask, unique_ravelled

register = functools.partial(gin.register, module="gtf.data")


@dataclass
class SemiSupervisedSingle:
    """Data class for a single sparsely labelled graph."""

    node_features: tp.Union[tf.Tensor, tf.SparseTensor]  # [N, F]
    adjacency: tf.SparseTensor  # [N, N]
    labels: tf.Tensor  # [N]
    train_ids: tf.Tensor  # [n_train << N]
    validation_ids: tf.Tensor  # [n_eval << N]
    test_ids: tf.Tensor  # [n_test < N]

    def __post_init__(self):
        self.node_features.shape.assert_has_rank(2)
        assert self.node_features.dtype.is_floating
        self.adjacency.shape.assert_has_rank(2)
        assert self.adjacency.shape[0] == self.adjacency.shape[1], self.adjacency.shape

        for ids in (self.train_ids, self.validation_ids, self.test_ids):
            ids.shape.assert_has_rank(1)
            assert ids.dtype.is_integer


class EdgeData(tp.NamedTuple):
    indices: tf.Tensor
    labels: tf.Tensor
    weights: tp.Optional[tf.Tensor]


class AutoencoderDataV2(tp.NamedTuple):
    features: tp.Optional[tf.Tensor]  # [num_nodes, num_features]
    adjacency: tf.SparseTensor  # [num_nodes, num_nodes]
    train_edges: EdgeData
    validation_edges: tp.Optional[EdgeData]
    test_edges: tp.Optional[EdgeData]


class AutoencoderData(tp.NamedTuple):
    features: tp.Optional[tf.Tensor]  # [num_nodes, num_features]
    adjacency: tf.SparseTensor  # [num_nodes, num_nodes]
    train_labels: tf.Tensor  # [num_nodes**2, 1]
    true_labels: tf.Tensor  # [num_nodes**2, 1]
    train_weights: tf.Tensor  # [num_nodes**2]
    validation_weights: tp.Optional[tf.Tensor]  # [num_nodes**2]
    test_weights: tp.Optional[tf.Tensor]  # [num_nodes**2]


def subgraph(single: SemiSupervisedSingle, indices: tf.Tensor):
    indices = tf.sort(tf.convert_to_tensor(indices, tf.int64))
    adj = single.adjacency
    adj = stfu.ops.gather(adj, indices, axis=0)
    adj = stfu.ops.gather(adj, indices, axis=1)
    values = tf.range(tf.size(indices, indices.dtype), dtype=indices.dtype)
    init = tf.lookup.KeyValueTensorInitializer(indices, values)
    default_value = -tf.ones((), indices.dtype)
    table = tf.lookup.StaticHashTable(init, default_value)

    def lookup(ids):
        ids = table.lookup(ids)
        return tf.boolean_mask(ids, ids != default_value)

    node_features = single.node_features
    if isinstance(node_features, tf.SparseTensor):
        node_features = stfu.ops.gather(node_features, indices, axis=0)
    else:
        node_features = (tf.gather(node_features, indices, axis=0),)

    return SemiSupervisedSingle(
        node_features,
        adj,
        tf.gather(single.labels, indices, axis=0),
        train_ids=lookup(single.train_ids),
        validation_ids=lookup(single.validation_ids),
        test_ids=lookup(single.test_ids),
    )


def get_largest_component(
    single: SemiSupervisedSingle, directed: bool = True, connection="weak"
):
    indices = get_largest_component_indices(
        single.adjacency, directed=directed, connection=connection
    )
    return subgraph(single, indices)


def get_largest_component_autoencoder(
    data: AutoencoderData, directed: bool = True, connection="weak"
) -> AutoencoderData:
    raise NotImplementedError("TODO")


def preprocess_weights(ids: tf.Tensor, num_nodes, normalize: bool = True):
    weights = indices_to_mask(ids, num_nodes, dtype=tf.float32)
    if normalize:
        weights = weights / tf.size(ids, out_type=tf.float32)
    return weights


TensorTransform = tp.Callable[[tf.Tensor], tf.Tensor]
SparseTensorTransform = tp.Callable[[tf.SparseTensor], tf.SparseTensor]


@register
def num_nodes(data: SemiSupervisedSingle) -> int:
    return data.adjacency.shape[0]


@register
def preprocess_base(
    data: SemiSupervisedSingle,
    largest_component_only: bool = False,
    features_transform: tp.Union[TensorTransform, tp.Iterable[TensorTransform]] = (),
    adjacency_transform: tp.Union[
        SparseTensorTransform, tp.Iterable[SparseTensorTransform]
    ] = (),
) -> SemiSupervisedSingle:
    if largest_component_only:
        data = get_largest_component(data)

    return SemiSupervisedSingle(
        transformed(data.node_features, features_transform),
        transformed(data.adjacency, adjacency_transform),
        data.labels,
        data.train_ids,
        data.validation_ids,
        data.test_ids,
    )


@register
def preprocess_classification_single(
    data: SemiSupervisedSingle,
    largest_component_only: bool = False,
    features_transform: tp.Union[TensorTransform, tp.Iterable[TensorTransform]] = (),
    adjacency_transform: tp.Union[
        SparseTensorTransform, tp.Iterable[SparseTensorTransform]
    ] = (),
) -> DataSplit:
    data = preprocess_base(
        data,
        largest_component_only=largest_component_only,
        features_transform=features_transform,
        adjacency_transform=adjacency_transform,
    )
    return to_classification_split(data)


@register
def to_classification_split(data: SemiSupervisedSingle) -> DataSplit:
    num_nodes = tf.shape(data.node_features, out_type=tf.int64)[0]

    def get_data(indices):
        weights = preprocess_weights(indices, num_nodes)
        labels = data.labels
        labels = tf.where(weights > 0, labels, tf.zeros_like(labels))
        example = ((data.node_features, data.adjacency), labels, weights)
        return (example,)

    return DataSplit(
        *(get_data(ids) for ids in (data.train_ids, data.validation_ids, data.test_ids))
    )


@register
def to_autoencoder_split(data: AutoencoderData) -> DataSplit:
    features = data.features
    adjacency = data.adjacency
    inputs = adjacency if features is None else (features, adjacency)

    def get_examples(labels, weights) -> tp.Optional[tp.Iterable]:
        if weights is None:
            return None
        assert labels is not None
        example = inputs, labels, weights
        return (example,)

    train_data = get_examples(data.train_labels, data.train_weights)
    assert train_data is not None
    return DataSplit(
        train_data,
        get_examples(data.true_labels, data.validation_weights),
        get_examples(data.true_labels, data.test_weights),
    )


@register
def transformed(base, transforms: tp.Union[tp.Callable, tp.Iterable[tp.Callable]]):
    if transforms is None:
        return base
    if callable(transforms):
        return transforms(base)
    for transform in transforms:
        base = transform(base)
    return base


def mask_test_edges(
    adj: sp.coo_matrix,
    seed: int = 0,
    validation_frac: float = 0.05,
    test_frac: float = 0.1,
    validation_edges_in_adj: bool = False,
):
    """
    Split edges for graph autoencoder into train/validation/test splits.

    Based on https://github.com/tkipf/gae/blob/master/gae/preprocessing.py

    Args:
        adj: scipy.sparse.coo_matrix adjacency matrix.
    """
    rng = np.random.default_rng(seed)

    def sparse_to_tuple(sparse_mx):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * validation_frac))

    all_edge_idx = list(range(edges.shape[0]))
    rng.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val : (num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]

    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # TODO: use sets?
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = rng.integers(0, adj.shape[0])
        idx_j = rng.integers(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = rng.integers(0, adj.shape[0])
        idx_j = rng.integers(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    if validation_edges_in_adj:
        adj_edges = np.concatenate((train_edges, val_edges), axis=0)
    else:
        adj_edges = train_edges

    data = np.ones(adj_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.coo_matrix((data, adj_edges.T), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return (
        adj_train,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    )


def _get_train_labels_and_weights(
    adj: tf.SparseTensor, *, remove_self_edges: bool = False
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    num_nodes = adj.shape[0]
    num_nodes2 = num_nodes ** 2
    num_edges = adj.indices.shape[0]
    train_weights = tf.ones((num_nodes2,), tf.float32)
    pos_weight = float(num_nodes2 - num_edges) / num_edges

    # train weights only accurate for loss
    adji = tf.sparse.add(adj, tf.sparse.eye(adj.shape[0]))
    adji = adji.with_values(tf.ones_like(adji.values, dtype=tf.bool))
    train_labels = tf.reshape(tf.sparse.to_dense(adji), (-1, 1))

    train_weights = tf.where(
        tf.squeeze(train_labels, -1), pos_weight * train_weights, train_weights
    ) / float((num_nodes2 - num_edges) * 2)
    if remove_self_edges:
        train_weights = tf.where(
            tf.reshape(tf.eye(num_nodes, dtype=bool), (-1,)),
            tf.zeros_like(train_weights),
            train_weights,
        )
    return train_labels, train_weights


def _as_1d(dense_shape, *edges):
    edges = np.concatenate(edges, axis=0)
    edges = np.ravel_multi_index(edges.T, dense_shape)
    edges.sort()
    return edges


def _sparse_to_labels(st: tf.SparseTensor):
    return tf.reshape(tf.cast(tf.sparse.to_dense(st), bool), (-1, 1))


def _edges_to_weights(dense_shape, *edges):
    ids = _as_1d(dense_shape, *edges)
    return preprocess_weights(ids, np.prod(dense_shape))


@register
def preprocess_autoencoder_data(
    data: SemiSupervisedSingle,
    largest_component_only: bool = False,
    features_transform: tp.Union[TensorTransform, tp.Iterable[TensorTransform]] = (),
    adjacency_transform: tp.Union[
        SparseTensorTransform, tp.Iterable[SparseTensorTransform]
    ] = (),
    seed: tp.Optional[int] = None,
    validation_frac: float = 0.05,
    test_frac: float = 0.1,
    validation_edges_in_adj: bool = False,
    remove_self_edges: bool = False,
) -> AutoencoderData:
    if seed is None:
        seed = tf.random.uniform(  # pylint: disable=unexpected-keyword-arg
            (), maxval=np.iinfo(np.int64).max, dtype=tf.int64
        ).numpy()
    assert seed is not None
    if largest_component_only:
        data = get_largest_component(data)
    adj = data.adjacency
    adj_sp = sp.coo_matrix((adj.values.numpy(), adj.indices.numpy().T), shape=adj.shape)
    (
        adj_train,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges(
        adj_sp,
        seed=seed,
        validation_frac=validation_frac,
        test_frac=test_frac,
        validation_edges_in_adj=validation_edges_in_adj,
    )
    adj_train = adj_train.tocoo()
    adj_train = tf.SparseTensor(
        np.stack((adj_train.row, adj_train.col), axis=1),
        tf.convert_to_tensor(adj_train.data, tf.float32),
        dense_shape=adj_train.shape,
    )
    node_features = transformed(data.node_features, features_transform)
    true_labels = tf.reshape(
        tf.convert_to_tensor(np.asarray(adj_sp.todense()), dtype=tf.bool), (-1, 1)
    )

    # validation/test weights only accurate for metrics
    validation_weights = _edges_to_weights(adj.shape, val_edges, val_edges_false)
    test_weights = _edges_to_weights(adj.shape, test_edges, test_edges_false)

    train_labels, train_weights = _get_train_labels_and_weights(
        adj_train, remove_self_edges=remove_self_edges
    )
    adj_train = transformed(adj_train, adjacency_transform)
    return AutoencoderData(
        node_features,
        adj_train,
        train_labels,
        true_labels,
        train_weights,
        validation_weights,
        test_weights,
    )


@register
def random_features(
    features_or_num_nodes: tp.Union[tf.Tensor, int], size: int,
):
    if isinstance(features_or_num_nodes, int):
        num_nodes = features_or_num_nodes
    else:
        num_nodes = features_or_num_nodes.shape[0]
    return tf.random.normal((num_nodes, size))


def _get_dir(data_dir: tp.Optional[str], environ: str, default: tp.Optional[str]):
    if data_dir is None:
        data_dir = os.environ.get(environ, default)
    if data_dir is None:
        return None
    return os.path.expanduser(os.path.expandvars(data_dir))


def _load_dgl_graph(dgl_example, make_symmetric=False) -> tf.SparseTensor:
    r, c = (x.numpy() for x in dgl_example.edges())
    shape = (dgl_example.num_nodes(),) * 2
    if make_symmetric:
        # add symmetric edges
        r = np.array(r, dtype=np.int64)
        c = np.array(c, dtype=np.int64)
        # remove diagonals
        valid = r != c
        r = r[valid]
        c = c[valid]
        r, c = np.concatenate((r, c)), np.concatenate((c, r))
        i1d = np.ravel_multi_index((r, c), shape)
        i1d = np.unique(i1d)  # also sorts
        r, c = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
            i1d, shape
        )
    return tf.SparseTensor(
        tf.stack((r, c), axis=1), tf.ones(r.size, dtype=tf.float32), shape
    )


@register
def citations_data(name: str = "cora") -> SemiSupervisedSingle:
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
    assert dataset.cardinality() == 1
    element = dataset.get_single_element()
    graph = element["graph"]
    links = graph["links"]
    features = graph["node_features"]
    labels = element["node_labels"]
    train_ids = element["train_ids"]
    validation_ids = element["validation_ids"]
    test_ids = element["test_ids"]

    adjacency = tf.SparseTensor(
        links,
        tf.ones((tf.shape(links)[0],), dtype=tf.float32),
        (tf.shape(features, tf.int64)[0],) * 2,
    )

    out = SemiSupervisedSingle(
        features, adjacency, labels, train_ids, validation_ids, test_ids
    )
    return out


@register
def ogbn_data(
    name: str, data_dir: tp.Optional[str] = None, make_symmetric: bool = True,
) -> SemiSupervisedSingle:
    import ogb.nodeproppred  # pylint: disable=import-outside-toplevel

    root_dir = _get_dir(data_dir, "OGB_DATA", "~/ogb")
    if not name.startswith("ogbn-"):
        name = f"ogbn-{name}"

    print(f"Loading dgl ogbn-{name}...")
    ds = ogb.nodeproppred.DglNodePropPredDataset(name, root=root_dir)
    print("Got base data. Initial preprocessing...")
    split_ids = ds.get_idx_split()
    train_ids, validation_ids, test_ids = (
        tf.convert_to_tensor(split_ids[n].numpy(), tf.int64)
        for n in ("train", "valid", "test")
    )
    example, labels = ds[0]
    feats = tf.convert_to_tensor(example.ndata["feat"], tf.float32)
    labels = labels.numpy().squeeze(1)
    labels[np.isnan(labels)] = -1
    labels = tf.convert_to_tensor(labels, tf.int64)
    graph = _load_dgl_graph(example, make_symmetric=make_symmetric)
    print("Finished initial preprocessing")

    data = SemiSupervisedSingle(
        feats, graph, labels, train_ids, validation_ids, test_ids
    )
    print("num (nodes, edges, features):")
    print(
        data.node_features.shape[0],
        data.adjacency.values.shape[0],
        data.node_features.shape[1],
    )
    return data


def pos_neg_to_edges(pos_indices, neg_indices, mean_weights=False):
    indices = tf.concat((pos_indices, neg_indices), 0)
    labels = tf.concat(
        (
            tf.ones((pos_indices.shape[0], 1), dtype=bool),
            tf.zeros((neg_indices.shape[0], 1), dtype=bool),
        ),
        0,
    )
    if mean_weights:
        n = labels.shape[0]
        weights = tf.fill((n,), tf.constant(1 / n, dtype=tf.float32))
    else:
        weights = None
    return EdgeData(indices, labels, weights)


@register
def asymproj_data_v2(
    name: str = "ca-AstroPh",
    features_fn: tp.Optional[tp.Callable[[int], tf.Tensor]] = None,
    adjacency_transform: tp.Union[
        SparseTensorTransform, tp.Iterable[SparseTensorTransform]
    ] = (),
) -> AutoencoderDataV2:
    example = tfds.load(f"asymproj/{name}", split="full").get_single_element()
    train_pos = example["train_pos"]
    train_neg = example["train_neg"]
    test_pos = example["test_pos"]
    test_neg = example["test_neg"]
    n = example["num_nodes"]

    train_edges = pos_neg_to_edges(train_pos, train_neg, mean_weights=False)
    test_edges = pos_neg_to_edges(test_pos, test_neg, mean_weights=False)
    validation_edges = None

    adjacency = tf.SparseTensor(train_pos, tf.ones((train_pos.shape[0],)), (n, n))
    if features_fn is None:
        features = None
    else:
        features = features_fn(n)
    adjacency = transformed(adjacency, adjacency_transform)
    return AutoencoderDataV2(
        features, adjacency, train_edges, validation_edges, test_edges
    )


@register
def asymproj_data(
    name: str = "ca-AstroPh",
    symmetric: bool = True,
    use_all_train_edges: bool = False,
    remove_train_self_edges: bool = False,
    features_fn: tp.Optional[tp.Callable[[int], tf.Tensor]] = None,
    adjacency_transform: tp.Union[
        SparseTensorTransform, tp.Iterable[SparseTensorTransform]
    ] = (),
) -> AutoencoderData:
    example = tfds.load(f"asymproj/{name}", split="full").get_single_element()
    train_pos = example["train_pos"]
    train_neg = example["train_neg"]
    test_pos = example["test_pos"]
    test_neg = example["test_neg"]
    n = example["num_nodes"]

    def to_symmetric(indices):
        indices.shape.assert_has_rank(2)
        assert indices.shape[1] == 2
        row, col = tf.unstack(indices, axis=1)
        row, col = tf.concat((row, col), 0), tf.concat((col, row), 0)
        indices = tf.stack((row, col), axis=1)
        indices, _ = unique_ravelled(indices, (n, n), axis=1)
        return indices

    def pos_to_sparse(indices):
        return tf.SparseTensor(indices, tf.ones((indices.shape[0],)), (n, n))

    if symmetric:
        # no point in making test_pos symmetric
        train_pos = to_symmetric(train_pos)

    adjacency = pos_to_sparse(train_pos)
    adjacency = tf.sparse.reorder(adjacency)  # pylint: disable=no-value-for-parameter
    if use_all_train_edges:
        train_labels, train_weights = _get_train_labels_and_weights(
            adjacency, remove_self_edges=remove_train_self_edges
        )
    else:
        if remove_train_self_edges:
            raise NotImplementedError("TODO")
        train_labels = _sparse_to_labels(adjacency)
        train_edges = _as_1d(adjacency.shape, train_pos, train_neg)
        train_weights = preprocess_weights(train_edges, adjacency.shape[0] ** 2)

    all_edges = tf.concat((train_pos, test_pos), 0)
    all_edges, _ = unique_ravelled(all_edges, (n, n), axis=1)
    all_adj = pos_to_sparse(all_edges)
    all_adj = tf.sparse.reorder(all_adj)  # pylint: disable=no-value-for-parameter
    true_labels = _sparse_to_labels(all_adj)

    test_edges = _as_1d(adjacency.shape, test_pos, test_neg)
    test_weights = preprocess_weights(test_edges, adjacency.shape[0] ** 2)

    adjacency = transformed(adjacency, adjacency_transform)

    return AutoencoderData(
        None if features_fn is None else features_fn(adjacency.shape[0]),
        adjacency,
        train_labels,
        true_labels,
        train_weights,
        None,
        test_weights,
    )


def split_edges_walk_pooling(
    adj: sp.coo_matrix,
    validation_frac: float = 0.05,
    test_frac: float = 0.1,
    seed: int = 0,
    validation_edges_in_adj: bool = False,
    practical_neg_sample: bool = True,
):
    """Split edges consistent with the method used in Walk Pooling."""
    rng = np.random.default_rng(seed)

    def random_perm(size, dtype=np.int64):
        perm = np.arange(size, dtype=dtype)
        rng.shuffle(perm)
        return perm

    row = adj.row
    col = adj.col
    mask = row < col
    row, col = row[mask], col[mask]
    num_edges = row.size
    num_nodes = adj.shape[0]
    n_v = int(validation_frac * num_edges)  # number of validation positive edges
    n_t = int(test_frac * num_edges)  # number of test positive edges
    # split positive edges
    perm = random_perm(num_edges)
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    val_pos = np.stack([r, c], axis=1)
    r, c = row[n_v : n_v + n_t], col[n_v : n_v + n_t]
    test_pos = np.stack([r, c], axis=1)
    r, c = row[n_v + n_t :], col[n_v + n_t :]
    train_pos = np.stack([r, c], axis=1)

    # sample negative edges
    if practical_neg_sample:
        neg_adj_mask = np.ones((num_nodes, num_nodes), dtype=bool)
        neg_adj_mask = np.triu(neg_adj_mask, 1)
        neg_adj_mask[row, col] = 0

        # Sample the test negative edges first
        neg_row, neg_col = np.where(neg_adj_mask)
        perm = random_perm(neg_row.size)[:n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        test_neg = np.stack([neg_row, neg_col], axis=1)

        # Sample the train and val negative edges with only knowing
        # the train positive edges
        row, col = train_pos.T
        neg_adj_mask = np.ones((num_nodes, num_nodes), dtype=bool)
        neg_adj_mask = np.triu(neg_adj_mask, 1)
        neg_adj_mask[row, col] = 0

        # Sample the train and validation negative edges
        neg_row, neg_col = np.where(neg_adj_mask)
        n_tot = n_v + train_pos.size
        perm = random_perm(neg_row.size)[:n_tot]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        val_neg = np.stack([row, col], axis=1)

        row, col = neg_row[n_v:], neg_col[n_v:]
        train_neg = np.stack([row, col], axis=1)

    else:
        # If practical_neg_sample == False, the sampled negative edges
        # in the training and validation set aware the test set

        neg_adj_mask = np.ones((num_nodes, num_nodes), dtype=bool)
        neg_adj_mask = np.triu(neg_adj_mask, 1)
        neg_adj_mask[row, col] = 0

        # Sample all the negative edges and split into val, test, train negs
        neg_row, neg_col = np.where(neg_adj_mask)
        perm = random_perm(neg_row.size)[: row.size]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        val_neg = np.stack([row, col], axis=1)

        row, col = neg_row[n_v : n_v + n_t], neg_col[n_v : n_v + n_t]
        test_neg = np.stack([row, col], axis=1)

        row, col = neg_row[n_v + n_t :], neg_col[n_v + n_t :]
        train_neg = np.stack([row, col], axis=1)

    pos = (
        np.concatenate((train_pos, val_pos), axis=0)
        if validation_edges_in_adj
        else train_pos
    )
    adj_train = sp.coo_matrix(
        (np.ones((pos.shape[0],), dtype=np.float32), pos.T), shape=adj.shape
    )
    adj_train = adj_train + adj_train.T
    return (
        adj_train,
        train_pos,
        train_neg,
        val_pos,
        val_neg,
        test_pos,
        test_neg,
    )


@register
def preprocess_autoencoder_data_v2(
    data: SemiSupervisedSingle,
    largest_component_only: bool = False,
    features_transform: tp.Union[TensorTransform, tp.Iterable[TensorTransform]] = (),
    adjacency_transform: tp.Union[
        SparseTensorTransform, tp.Iterable[SparseTensorTransform]
    ] = (),
    seed: tp.Optional[int] = None,
    validation_frac: float = 0.05,
    test_frac: float = 0.1,
    validation_edges_in_adj: bool = False,
    practical_neg_sample: bool = True,
) -> AutoencoderDataV2:
    if seed is None:
        seed = tf.random.uniform(  # pylint: disable=unexpected-keyword-arg
            (), maxval=np.iinfo(np.int64).max, dtype=tf.int64
        ).numpy()
    assert seed is not None
    if largest_component_only:
        data = get_largest_component(data)
    adj = data.adjacency
    adj_sp = sp.coo_matrix((adj.values.numpy(), adj.indices.numpy().T), shape=adj.shape)
    (
        adj_train,
        train_pos,
        train_neg,
        val_pos,
        val_neg,
        test_pos,
        test_neg,
    ) = split_edges_walk_pooling(
        adj_sp,
        seed=seed,
        validation_frac=validation_frac,
        test_frac=test_frac,
        validation_edges_in_adj=validation_edges_in_adj,
        practical_neg_sample=practical_neg_sample,
    )
    adj_train = adj_train.tocoo()
    adj_train = tf.SparseTensor(
        np.stack((adj_train.row, adj_train.col), axis=1),
        tf.convert_to_tensor(adj_train.data, tf.float32),
        dense_shape=adj_train.shape,
    )

    return AutoencoderDataV2(
        transformed(data.node_features, features_transform),
        transformed(adj_train, adjacency_transform),
        pos_neg_to_edges(train_pos, train_neg, True),
        pos_neg_to_edges(val_pos, val_neg, True),
        pos_neg_to_edges(test_pos, test_neg, True),
    )

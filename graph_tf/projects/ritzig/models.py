from typing import Callable, Optional, Union

import gin
import tensorflow as tf

from graph_tf.data.single import SemiSupervisedSingle
from graph_tf.utils.ops import to_laplacian


def col_normalize(node_features: Union[tf.Tensor, tf.SparseTensor]):
    if isinstance(node_features, tf.SparseTensor):
        node_features = tf.sparse.to_dense(node_features)
    mean, std = tf.nn.moments(node_features, axes=[0], keepdims=True)
    return tf.where(
        std == 0, tf.zeros_like(node_features), (node_features - mean) / std
    )


@gin.configurable(module="gtf.ritzig")
def gaussian_similarity(n0, n1, normalize=True):
    dist2 = tf.math.squared_difference(n0, n1)
    if normalize:
        dist2 = dist2 / tf.cast(tf.shape(n0)[0], dist2.dtype)
    return tf.exp(-tf.reduce_sum(dist2, axis=-1))


@gin.configurable(module="gtf.ritzig")
def cosine_similarity(n0, n1):
    return tf.einsum("ij,ij->i", n0, n1)
    # return tf.keras.backend.dot(n0, n1)


@gin.configurable(module="gtf.ritzig")
def all_same_similarity(n0, n1):
    del n1
    return tf.ones((tf.shape(n0)[0],), dtype=n0.dtype)


@gin.configurable(module="gtf.ritzig")
def preprocess_single(
    data: SemiSupervisedSingle,
    features_fn: Callable = col_normalize,
    edge_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
    normalize: bool = True,
):
    node_features = data.node_features
    num_nodes = tf.shape(node_features, out_type=tf.int64)[0]
    node_features = features_fn(node_features)

    edges = data.edges
    num_edges = tf.shape(edges, out_type=tf.int64)[0]
    if edge_fn is None:
        weights = tf.ones((num_edges,), dtype=tf.float32)
    else:
        weights = edge_fn(*tf.unstack(tf.gather(node_features, edges, axis=0), axis=1))
    adj = tf.SparseTensor(edges, weights, (num_nodes, num_nodes))
    L = to_laplacian(adj, normalize=normalize)

    print("Computing eigenvectors...")
    vals, vecs = tf.linalg.eigh(tf.sparse.to_dense(L))
    print("-----------------")
    print(vals.numpy())
    print("-----------------")
    node_features = vecs

    def get_data(ids):
        return (
            tf.gather(node_features, ids, axis=0),
            tf.gather(data.labels, ids, axis=0),
        )

    return (
        get_data(data.train_ids),
        get_data(data.validation_ids),
        get_data(data.test_ids),
    )


# @gin.configurable(module="gtf.ritzig")
# def preprocess_single(
#     data: SemiSupervisedSingle,
#     features_fn: Callable = col_normalize,
#     edge_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = cosine_similarity,
#     edge_bias: float = 0.0,
#     normalize: bool = True,
#     # subspace_dim: int = 128,
# ):
#     node_features = data.node_features
#     num_nodes = tf.shape(node_features, out_type=tf.int64)[0]
#     node_features = features_fn(node_features)

#     # pca = PCA(n_components=256)
#     # # node_features = pca.fit_transform(node_features.numpy().T)
#     # # # node_features = tf.constant(node_features.T, dtype=tf.float32)
#     # node_features = pca.fit_transform(node_features.numpy())
#     # node_features = tf.constant(node_features, dtype=tf.float32)
#     # s, u, v = tf.linalg.svd(node_features)
#     # node_features = u[:, :32]
#     # node_features = u

#     # tf.debugging.assert_non_negative(
#     #     node_features, message="node features must be non-negative"
#     # )
#     tf.debugging.assert_all_finite(node_features, "some node_features non-finite")
#     i, j = tf.unstack(data.edges, axis=-1)
#     edge_weights = edge_fn(
#         tf.gather(node_features, i, axis=0), tf.gather(node_features, j, axis=0)
#     )
#     if edge_bias:
#         edge_weights = edge_weights + edge_bias
#     tf.debugging.assert_non_negative(
#         edge_weights, message="edge_weights must be non-negative"
#     )
#     tf.debugging.assert_all_finite(edge_weights, message="edge_weights must be finite")
#     diag_indices = tf.tile(tf.expand_dims(tf.range(num_nodes), axis=1), (1, 2))
#     diag_values = tf.math.segment_sum(edge_weights, i)
#     if normalize:
#         factor = tf.where(
#             diag_values > 0, diag_values ** -0.5, tf.zeros_like(diag_values)
#         )
#         edge_weights = (
#             edge_weights * tf.gather(factor, i, axis=0) * tf.gather(factor, j, axis=0)
#         )
#         diag_values = tf.ones((num_nodes,), dtype=tf.float32)

#     dense_shape = (num_nodes, num_nodes)
#     neg_A = tf.SparseTensor(data.edges, -edge_weights, dense_shape)
#     D = tf.SparseTensor(diag_indices, diag_values, dense_shape)
#     L = tf.sparse.add(D, neg_A)

#     L = tf.sparse.to_dense(L)
#     vals, vecs = tf.linalg.eigh(L)
#     del vals
#     vecs = vecs[:, 1:]
#     node_features = vecs
#     # node_features = tf.concat((node_features, vecs), axis=-1)

#     # x = tf.random.Generator.from_seed(0).uniform(
#     #     (num_nodes, 1), minval=-0.5, maxval=0.5
#     # )

#     # Q, d, l = lanczos_iteration(L, x, subspace_dim + 1)
#     # Q = tf.squeeze(Q[:, :-1], axis=-1)
#     # d = tf.squeeze(d, axis=-1)
#     # l = tf.squeeze(l[:-1], axis=-1)

#     # diags = tf.stack(
#     #     (
#     #         tf.pad(l, [[0, 1]]),  # pylint:disable=no-value-for-parameter
#     #         d,
#     #         tf.pad(l, [[1, 0]]),  # pylint:disable=no-value-for-parameter
#     #     ),
#     #     axis=0,
#     # )
#     # td = tf.linalg.LinearOperatorTridiag(diags, is_self_adjoint=True)

#     # vals, vecs = tf.linalg.eigh(td.to_dense())
#     # # # ensure first eigenvalue is close to zero
#     # # v0 = vals[0]
#     # # tf.debugging.assert_less(v0, 1e-3)
#     # # tf.debugging.assert_greater(v0, -1e-3)
#     # # # remove first eigenvector, which should be close to constant
#     # vecs = vecs[:, 1:]

#     # ritz_features = tf.matmul(Q, vecs)
#     # node_features = ritz_features
#     # # node_features = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
#     # #     [node_features, ritz_features], axis=-1
#     # # )

#     def get_data(ids):
#         return (
#             tf.gather(node_features, ids, axis=0),
#             tf.gather(data.labels, ids, axis=0),
#         )

#     return (
#         get_data(data.train_ids),
#         get_data(data.validation_ids),
#         get_data(data.test_ids),
#     )

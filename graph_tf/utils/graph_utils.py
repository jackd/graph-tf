import inspect
import typing as tp

import scipy.sparse as sp
import stfu
import tensorflow as tf

from graph_tf.utils.linalg import SparseLinearOperator, SubtractProjection
from graph_tf.utils.random_utils import as_rng
from graph_tf.utils.scipy_utils import to_scipy


def get_largest_component_indices(
    adjacency: tf.SparseTensor,
    *,
    directed: bool = True,
    connection="weak",
) -> tf.Tensor:
    """
    Get the indices associated with the largest connected component.

    Args:
        adjacency: [n, n]adjacency matrix
        directed, connection: used in get_component_labels

    Returns:
        [size], int64 indices in [0, n) of nodes in the largest connected component,
            size <= n.
    """
    nc, labels = get_component_labels(  # pylint: disable=unpacking-non-sequence
        adjacency, directed=directed, connection=connection
    )
    if nc == 1:
        return tf.range(adjacency.shape[0], dtype=tf.int64)
    sizes = tf.math.unsorted_segment_sum(tf.ones_like(labels), labels, nc)
    indices = tf.squeeze(tf.where(labels == tf.argmax(sizes)), axis=1)
    return indices


class ComponentLabels(tp.NamedTuple):
    num_components: int
    labels: tf.Tensor


def get_component_labels(
    adjacency: tf.SparseTensor,
    dtype: tf.DType = tf.int64,
    *,
    directed: bool = True,
    connection="weak",
) -> ComponentLabels:
    """
    Get labels for each connected components based on scipy implementation.

    Args:
        adjacency: [n, n] graph adjacency. Values are ignored.
        dtype: dtype of the returned labels.
        directed, connection: passed to `scipy.sparse.csgraph.connected_components`.

    Returns:
        num_components: number of connected components
        labels: [n] labels in [0, num_components) and the given dtype.
    """
    adjacency = to_scipy(adjacency).tocsr()
    ncomponents, labels = sp.csgraph.connected_components(
        adjacency, return_labels=True, directed=directed, connection=connection
    )
    return ComponentLabels(ncomponents, tf.convert_to_tensor(labels, dtype=dtype))


def get_laplacian_zero_eigs(
    adjacency: tf.SparseTensor,
    symmetric_normalized: bool = False,
    normalize: bool = True,
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    """
    Get eigenvectors of eigenvalue 0 of the associated Laplacian.

    If symmetric_normalized is True, the returned eigenvectors are for the
    symmetric-normalized Laplacian, `x_i prop_to sqrt(d_i)`, otherwise
    `x_i prop_to ones_i`.

    Args:
        adjacency: [n, n] adjacency matrix
        symmetric_normalized:
        normalize: if True, the result is column-normalized

    Returns:
        X: [n, num_components]
    """
    dtype = tf.dtypes.as_dtype(dtype)
    assert dtype.is_floating, dtype
    nc, labels = get_component_labels(  # pylint: disable=unpacking-non-sequence
        adjacency, directed=False
    )
    if symmetric_normalized:
        values = tf.sparse.reduce_sum(adjacency, axis=1)
        values = tf.cast(values, dtype)
        values = tf.sqrt(values)
    else:
        values = tf.ones((adjacency.dense_shape[0],), dtype=dtype)
    z = tf.zeros_like(values)
    x_cols = [tf.where(labels == i, values, z) for i in range(nc)]
    X = tf.stack(x_cols, axis=1)
    if normalize:
        X = X / tf.linalg.norm(X, axis=0, keepdims=True)
    return X


def laplacian(x: tf.SparseTensor) -> tf.SparseTensor:
    d = tf.sparse.reduce_sum(x, axis=0)
    return tf.sparse.add(stfu.diag(d), x.with_values(-x.values))


def signed_incidence(adj: tf.SparseTensor) -> tf.SparseTensor:
    """
    Args:
        adj: [n, n] unweighted adjacency matrix.
        dtype:

    Returns:
        B: [m, n] such that `B.T @ B = L`, where m is the number of unique edges.
    """
    num_nodes = adj.dense_shape[0]
    i, j, v = tril_indices(adj, return_values=True)
    v = tf.sqrt(v)
    num_edges = tf.shape(i, tf.int64)[0]
    edge_indices = tf.range(num_edges, dtype=tf.int64)
    pos_indices = tf.stack((edge_indices, i), axis=1)
    neg_indices = tf.stack((edge_indices, j), axis=1)
    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    indices = tf.concat((pos_indices, neg_indices), axis=0)
    values = tf.concat(
        (v, -v),
        axis=0,
    )
    # pylint: enable=unexpected-keyword-arg,no-value-for-parameter
    incidence = tf.SparseTensor(indices, values, (num_edges, num_nodes))
    incidence = tf.sparse.reorder(incidence)  # pylint: disable=no-value-for-parameter
    return incidence


def tril(x: tf.SparseTensor) -> tf.SparseTensor:
    i, j, v = tril_indices(x, return_values=True)
    return tf.SparseTensor(tf.stack((i, j), axis=1), v, x.dense_shape)


def tril_indices(x: tf.SparseTensor, return_values: bool = False):
    i, j = tf.unstack(x.indices, axis=1)
    mask = i > j
    i, j = (tf.boolean_mask(ind, mask) for ind in (i, j))
    if return_values:
        return i, j, tf.boolean_mask(x.values, mask)
    return i, j


def effective_resistance_z(adj: tf.SparseTensor, **cg_kwargs) -> tf.Tensor:
    """
    Get the effective resistance matrix Z.

    The effective resistance matrix Z allows the effective resistance to be

    effective_resistance(i, j) = tf.reduce_sum((Z[i] - Z[j])**2, axis=1)

    Args:
        adj: adjacency matrix.
        **cg_kwargs: kwargs passed to `tf.linalg.experimental.conjugate_gradient`

    Returns:
        Z: [num_nodes, num_edges]
    """
    U = signed_incidence(adj)  # [m, n]
    U = tf.sparse.to_dense(U)
    L = laplacian(adj)
    X = get_laplacian_zero_eigs(adj, symmetric_normalized=False, normalize=True)

    A = tf.linalg.LinearOperatorComposition(
        (SparseLinearOperator(L), SubtractProjection(X)),
        is_self_adjoint=True,
        is_positive_definite=True,
        is_square=True,
    )

    max_iter = cg_kwargs.pop(
        "max_iter",
        inspect.signature(tf.linalg.experimental.conjugate_gradient)
        .parameters["max_iter"]
        .default,
    )

    def map_fn(x):
        sol = tf.linalg.experimental.conjugate_gradient(
            A, x, max_iter=max_iter, **cg_kwargs
        )
        tf.debugging.assert_less(sol.i, max_iter)
        return sol.x

    ZT = tf.vectorized_map(map_fn, U)  # [m, n]
    return tf.transpose(ZT, (1, 0))  # [n, m]


def approx_effective_resistance_z(
    adj: tf.SparseTensor,
    jl_factor: float = 4.0,
    rng: tp.Union[int, tf.random.Generator] = 0,
    **cg_kwargs,
) -> tf.Tensor:
    """
    Get the approximate effective resistance matrix Z.

    The effective resistance matrix Z allows the effective resistance to be approximated
    by

    effective_resistance(i, j) = tf.reduce_sum((Z[i] - Z[j])**2, axis=1)

    The approximation comes from performing random projections.

    Implementation based on that provided in Laplacians.jl.

    https://github.com/danspielman/Laplacians.jl/blob/master/src/sparsify.jl#L37

    Args:
        adj: adjacency matrix.
        jl_factor: Johnson-Lindenstrauss dimension factor to scale the number of random
            projection vectors. The higher the value the more accurate the
            approximation.

    Returns:
        Z: [num_nodes, k], where `k = int(jl_factor * log(num_nodes))`
    """
    rng: tf.random.Generator = as_rng(rng)
    n = adj.dense_shape[0]
    U = signed_incidence(adj)  # [m, n]
    m = U.dense_shape[0]
    k = tf.cast(jl_factor * tf.math.log(tf.cast(n, tf.float32)), tf.int64)

    L = laplacian(adj)
    X = get_laplacian_zero_eigs(adj, symmetric_normalized=False, normalize=True)
    A = tf.linalg.LinearOperatorComposition(
        (SparseLinearOperator(L), SubtractProjection(X)),
        is_self_adjoint=True,
        is_positive_definite=True,
        is_square=True,
    )

    # NOTE: the stddev arg here means the results are approximately row-normal
    R = rng.normal((m, k), stddev=tf.math.rsqrt(tf.cast(k, tf.float32)))

    UTR = tf.sparse.sparse_dense_matmul(U, R, adjoint_a=True)  # [n, k]
    RTU = tf.transpose(UTR)  # [k, n]
    max_iter = cg_kwargs.pop(
        "max_iter",
        inspect.signature(tf.linalg.experimental.conjugate_gradient)
        .parameters["max_iter"]
        .default,
    )

    def map_fn(x):
        sol = tf.linalg.experimental.conjugate_gradient(
            A, x, max_iter=max_iter, **cg_kwargs
        )
        tf.debugging.assert_less(sol.i, max_iter)
        return sol.x

    ZT = tf.vectorized_map(map_fn, RTU)  # [k, n]
    return tf.transpose(ZT, (1, 0))  # [n, k]


def get_effective_resistance(Z: tf.Tensor, i: tf.Tensor, j: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(
        tf.math.squared_difference(tf.gather(Z, i, axis=0), tf.gather(Z, j, axis=0)),
        axis=-1,
    )


def get_pairwise_effective_resistance(Z: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """
    Get the pairwise effective resistances.

    This implementation is equivalent to but more efficient than
    `get_effective_resistance(Z, *tf.meshgrid(indices, indices, indexing='ij'))`.

    Rather than computing the difference (Zi - Zj) then computing the norm (which would
    require O(d^2 k) space and time), we use `|Zi|^2 + |Zj|^2 - 2 * Zi @ Zj.T` which is
    O(d^2) in space and O(d^2 k) in time.

    Args:
        Z: [n, k] float effective resistance matrix as returned by
            `[approx_]effective_resistance_z`.
        indices: [d] int relevant indices in [0, n).

    Returns:
        resistances: [d, d] float matrix of effective resistances.
    """
    assert indices.dtype.is_integer
    assert Z.dtype.is_floating
    indices.shape.assert_has_rank(1)
    Z.shape.assert_has_rank(2)

    Zi = tf.gather(Z, indices, axis=0)
    n2 = tf.reduce_sum(tf.square(Zi), axis=1)
    return n2 + tf.expand_dims(n2, 1) - 2 * tf.linalg.matmul(Zi, Zi, adjoint_b=True)

import functools
import typing as tp

import gin
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf
from tflo.extras import LinearOperatorSparseMatrix

SparseLinearOperator = LinearOperatorSparseMatrix

register = functools.partial(gin.register, module="gtf.utils.linalg")


class EigenDecomposition(tp.NamedTuple):
    values: tf.Tensor  # [K]
    vectors: tf.Tensor  # [N, K]


@register
def spectral_features(adj: tf.SparseTensor, k: int, **kwargs):
    return laplacian_eigsh_from_adjacency(adj, k=k, **kwargs).vectors


def sparse_tensor_to_coo(X: tf.SparseTensor) -> sp.coo_matrix:
    if sp.issparse(X):
        return X.tocoo()
    return sp.coo_matrix((X.values.numpy(), X.indices.numpy().T), shape=X.shape)


def eigsh(X: tf.SparseTensor, k: int = 6, **kwargs) -> EigenDecomposition:
    X_coo = sparse_tensor_to_coo(X)
    w, v = la.eigsh(X_coo, k=k, **kwargs)
    return EigenDecomposition(tf.convert_to_tensor(w), tf.convert_to_tensor(v))


def laplacian_eigsh_from_adjacency_np(
    adj: sp.spmatrix, k: int = 6, **kwargs
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """`numpy` interface to `laplacian_eigsh_from_adjacency`."""
    if "which" in kwargs:
        assert kwargs["which"] == "SM", kwargs["which"]
        del kwargs["which"]

    d = np.asarray(adj.sum(axis=1)).squeeze(axis=1)
    d_sqrt = np.sqrt(d)
    adj = sp.coo_matrix(
        (adj.data / (d_sqrt[adj.row] * d_sqrt[adj.col]), (adj.row, adj.col)),
        shape=adj.shape,
    )
    shifted_lap = -sp.eye(adj.shape[0], dtype=adj.dtype) - adj
    w, v = la.eigsh(shifted_lap, k=k, v0=d_sqrt / d.sum(), **kwargs)
    w += 2
    return w, v


def laplacian_eigsh_from_adjacency(
    adj: tf.SparseTensor, k: int = 6, **kwargs
) -> EigenDecomposition:
    """
    Get partial eigenvalues/vectors of the symmetrically normalized laplacian.

    Args:
        adj: adjacency
        k: number of eigenpairs to extract.
        **kwargs: parsed to `scipy.sparse.linalg.cg`. If `which` is provided, it must be
            "SM".

    Returns:
        w: [k] smallest eigenvalues.
        v: [num_nodes, k] associated eigenvectors.
    """
    adj = sparse_tensor_to_coo(adj)
    w, v = laplacian_eigsh_from_adjacency_np(sparse_tensor_to_coo(adj), k=k, **kwargs)
    return EigenDecomposition(tf.convert_to_tensor(w), tf.convert_to_tensor(v))


def eigsh_lap(
    lap: tf.SparseTensor, k: int = 6, *, largest_kwargs=None, **kwargs
) -> EigenDecomposition:
    """
    Compute smallest `k` eigenvalues/vectors for a laplacian matrix.

    There are some issues using eigsh with `which="SM"` (smallest magnitude). Since we
    know Laplacian eigenvalues are all >= 0, we compute the largest eigenvalue `w_max`
    then compute the largest magnitude eigenvalues/vectors of `lap - w_max*I`.
    """
    X_coo = sp.coo_matrix((lap.values.numpy(), lap.indices.numpy().T), shape=lap.shape)
    w_max, _ = la.eigsh(X_coo, k=1, which="LM", **(largest_kwargs or {}))
    X_coo = X_coo - sp.eye(
        X_coo.shape[0], dtype=lap.dtype.as_numpy_dtype()
    ) * w_max.squeeze(0)
    w, v = la.eigsh(X_coo, k=k, which="LM", **kwargs)
    w += w_max
    return EigenDecomposition(tf.convert_to_tensor(w), tf.convert_to_tensor(v))


class SubtractProjection(tf.linalg.LinearOperator):  # pylint: disable=abstract-method
    """(I - X @ X.T) for X.shape == (n, r), r << n."""

    def __init__(self, X: tf.Tensor, **kwargs):
        X = tf.convert_to_tensor(X)
        X.shape.assert_has_rank(2)
        self._X = X
        super().__init__(
            dtype=X.dtype, is_self_adjoint=True, is_non_singular=False, **kwargs
        )

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        del adjoint  # always self-adjoint
        return x - tf.matmul(
            self._X, tf.matmul(self._X, x, adjoint_a=True, adjoint_b=adjoint_arg)
        )

    def _shape(self):
        return tf.TensorShape((self._X.shape[0],) * 2)

    def _shape_tensor(self):
        return tf.tile(tf.expand_dims(tf.shape(self._X, tf.int64)[0], 0), (2,))

    def _to_dense(self):
        return tf.eye(self._X.shape[0]) - tf.matmul(self._X, self._X, transpose_b=True)

    def _diag_part(self):
        return 1 - tf.reduce_sum(self._X**2, axis=1)

    def _shape_invariant_to_type_spec(self, shape):
        return tf.TensorSpec(shape, dtype=self.dtype)


# class SparseLinearOperator(tf.linalg.LinearOperator):  # pylint: disable=abstract-method
#     """`tf.linalg.LinearOperator` wrapper around `tf.SparseTensor`."""

#     def __init__(self, st: tf.SparseTensor, **kwargs):
#         self._st = st
#         super().__init__(dtype=self._st.dtype, **kwargs)

#     def _matmul(self, x, adjoint=False, adjoint_arg=False):
#         return tf.sparse.sparse_dense_matmul(
#             self._st, x, adjoint_a=adjoint, adjoint_b=adjoint_arg
#         )

#     def _shape(self):
#         return self._st.shape

#     def _shape_tensor(self):
#         return self._st.dense_shape

#     def _to_dense(self):
#         return tf.sparse.to_dense(self._st)

#     def _diag_part(self):
#         *batch_indices, i, j = tf.unstack(self._st.indices, axis=1)
#         mask = i == j
#         d = tf.boolean_mask(i, mask)

#         values = tf.boolean_mask(self._st.values, mask)
#         batch_dims, trailing_dims = tf.split(  # pylint: disable=no-value-for-parameter
#             self._st.dense_shape, [-1, 2]
#         )
#         # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
#         out_shape = tf.concat(
#             (batch_dims, tf.expand_dims(tf.reduce_min(trailing_dims), 0)), axis=0
#         )
#         # pylint: enable=unexpected-keyword-arg,no-value-for-parameter
#         return tf.scatter_nd(tf.stack((*batch_indices, d), axis=1), values, out_shape)

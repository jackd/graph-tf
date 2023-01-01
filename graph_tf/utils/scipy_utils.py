import typing as tp

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf
from scipy.sparse.linalg.eigen.arpack import eigsh


def normalize_sparse(A: sp.spmatrix) -> sp.spmatrix:
    """Get (D**-0.5) * A * (D ** -0.5), where D is the diagonalized row sum."""
    A = sp.coo_matrix(A)
    A.eliminate_zeros()
    rowsum = np.array(A.sum(1))
    assert np.all(rowsum >= 0)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
    # return A.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def chebyshev_polynomials(A: sp.spmatrix, k: int) -> tp.Sequence[sp.spmatrix]:
    """
    Calculate Chebyshev polynomials up to order k.

    Args:
        A: input sparse matrix
        k: order of chebyshev polynomial

    Returns:
        k+1 sparse matrices
    """

    A = normalize_sparse(A)
    N = A.shape[0]
    laplacian = sp.eye(N) - A
    largest_eigval, _ = eigsh(laplacian, 1, which="LM")
    scaled_laplacian = (2.0 / largest_eigval[0]) * laplacian - sp.eye(N)

    t_k = [sp.eye(N), scaled_laplacian]
    rescaled_laplacian = (2 * scaled_laplacian).tocsr(copy=False)

    for _ in range(2, k + 1):
        t_k.append(rescaled_laplacian.dot(t_k[-1]) - t_k[-2])

    return [t.tocoo(copy=False) for t in t_k]


def to_scipy(st: tf.SparseTensor) -> sp.coo_matrix:
    st.shape.assert_has_rank(2)
    assert tf.executing_eagerly()
    return sp.coo_matrix(
        (st.values.numpy(), st.indices.numpy().T), shape=st.dense_shape.numpy()
    )


def to_tf(sp_matrix: tp.Union[sp.coo_matrix, sp.csr_matrix]) -> tf.SparseTensor:
    coo: sp.coo_matrix = sp_matrix.tocoo()
    row, col = (tf.convert_to_tensor(x, tf.int64) for x in (coo.row, coo.col))
    return tf.SparseTensor(tf.stack((row, col), axis=1), coo.data, coo.shape)


class ShiftedLinearOperator(la.LinearOperator):
    """Shifts eigenvalues with vectors `u` to `old_eigenvalue + scalar`."""

    def __init__(self, op: la.LinearOperator, u: np.ndarray, scalar: float):
        self.op = la.aslinearoperator(op)
        self.u = u
        self.scalar = scalar
        assert op.shape == (u.shape[0], u.shape[0]), (op.shape, u.shape[0])
        assert u.ndim == 2, u.shape
        super().__init__(shape=op.shape, dtype=op.dtype)

    def _matvec(self, x):
        return self.op.matvec(x) + self.scalar * self.u @ (self.u.T @ x)

    def _matmul(self, x):
        return self.op.matmul(x) + self.scalar * self.u @ (self.u.T @ x)

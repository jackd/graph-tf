from typing import Sequence

import numpy as np
import scipy.sparse as sp
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


def chebyshev_polynomials(A: sp.spmatrix, k: int) -> Sequence[sp.spmatrix]:
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

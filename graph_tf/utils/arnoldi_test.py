import numpy as np
import tensorflow as tf

from graph_tf.utils.arnoldi import arnoldi_iteration as arnoldi_iteration_tf
from graph_tf.utils.test_utils import random_sparse


def arnoldi_iteration_np(A, b, n: int):
    """Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m × m array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1

    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.
    """
    m = A.shape[0]
    h = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1))
    q = b / np.linalg.norm(b)  # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector

    for k in range(n):
        v = A.dot(q)  # Generate a new candidate vector
        for j in range(k + 1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-12  # If v is shorter than this threshold it is the zero vector
        if h[k + 1, k] > eps:  # Add the produced vector to the list, unless
            q = v / h[k + 1, k]  # the zero vector is produced.
            Q[:, k + 1] = q
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h


class ArnoldiTest(tf.test.TestCase):
    def test_dense_consistent_with_numpy(self):
        m = 10
        n = 3

        rng = tf.random.Generator.from_seed(0)
        A = rng.normal((m, m))
        b = rng.normal((m, 1))

        Q_tf, h_tf = arnoldi_iteration_tf(A, b, n)
        Q_np, h_np = arnoldi_iteration_np(A.numpy(), tf.squeeze(b, 1).numpy(), n)

        self.assertAllClose(tf.squeeze(Q_tf, axis=-1), Q_np)
        self.assertAllClose(tf.squeeze(h_tf, axis=-1), h_np)

    def test_sparse_consistent_with_numpy(self):
        m = 100
        nnz = m * 10
        n = 10

        rng = tf.random.Generator.from_seed(0)
        A = random_sparse((m, m), nnz, rng=rng)
        b = rng.normal((m, 1))

        Q_tf, h_tf = arnoldi_iteration_tf(A, b, n)
        Q_np, h_np = arnoldi_iteration_np(
            tf.sparse.to_dense(A).numpy(), tf.squeeze(b, 1).numpy(), n
        )

        self.assertAllClose(tf.squeeze(Q_tf, axis=-1), Q_np)
        self.assertAllClose(tf.squeeze(h_tf, axis=-1), h_np)

    def test_dense_consistent_when_stacked(self):
        m = 10
        n = 3
        p = 5

        rng = tf.random.Generator.from_seed(0)
        A = rng.normal((m, m))
        b = rng.normal((m, p))

        bs = tf.split(  # pylint: disable=no-value-for-parameter,redundant-keyword-arg
            b, p, axis=-1
        )

        Q_stacked, h_stacked = arnoldi_iteration_tf(A, b, n)
        Q_sep, h_sep = zip(*(arnoldi_iteration_tf(A, bi, n) for bi in bs))
        Q_sep = tf.concat(  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            Q_sep, axis=-1
        )
        h_sep = tf.concat(  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            h_sep, axis=-1
        )

        self.assertAllClose(Q_stacked, Q_sep)
        self.assertAllClose(h_stacked, h_sep)

    def test_sparse_consistent_when_stacked(self):
        m = 100
        nnz = m * 10
        n = 10
        p = 5

        rng = tf.random.Generator.from_seed(0)
        A = random_sparse((m, m), nnz, rng=rng)
        b = rng.normal((m, p))

        bs = tf.split(  # pylint: disable=no-value-for-parameter,redundant-keyword-arg
            b, p, axis=-1
        )

        Q_stacked, h_stacked = arnoldi_iteration_tf(A, b, n)
        Q_sep, h_sep = zip(*(arnoldi_iteration_tf(A, bi, n) for bi in bs))
        Q_sep = tf.concat(  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            Q_sep, axis=-1
        )
        h_sep = tf.concat(  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            h_sep, axis=-1
        )

        self.assertAllClose(Q_stacked, Q_sep)
        self.assertAllClose(h_stacked, h_sep)

    def test_symmetric_case(self):
        m = 100
        nnz = m * 10
        n = 20
        p = 5

        rng = tf.random.Generator.from_seed(0)
        A = random_sparse((m, m), nnz, rng=rng)
        A = tf.sparse.add(A, tf.sparse.transpose(A, (1, 0)))  # make symmetric
        b = rng.normal((m, p))

        Qs, hs = arnoldi_iteration_tf(A, b, n, symmetric=True)

        Q, h = arnoldi_iteration_tf(A, b, n)
        self.assertAllClose(Q, Qs, atol=1e-5)
        self.assertAllClose(h, hs, atol=1e-5)


if __name__ == "__main__":
    tf.test.main()

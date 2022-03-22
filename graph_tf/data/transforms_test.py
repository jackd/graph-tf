import tensorflow as tf

from graph_tf.data import transforms
from graph_tf.utils import test_utils


class TransformsTest(tf.test.TestCase):
    def test_signed_incidence(self, seed: int = 0, n: int = 10, m: int = 50):
        a = test_utils.random_sparse((n, n), m, tf.random.Generator.from_seed(seed))
        adj_tril = transforms.tril(a)
        del a
        adj_tril = tf.sparse.map_values(tf.math.abs, adj_tril)
        adj = tf.sparse.reorder(  # pylint: disable=no-value-for-parameter
            tf.sparse.add(adj_tril, tf.sparse.transpose(adj_tril))
        )
        B = tf.sparse.to_dense(transforms.signed_incidence(adj))
        L0 = tf.matmul(tf.transpose(B), B)
        L1 = tf.sparse.to_dense(transforms.laplacian(adj))
        self.assertAllClose(L0, L1)

    def test_effective_resistance(self):
        # build an n-ring graph
        n = 10
        r = tf.range(n, dtype=tf.int64)
        i = tf.math.mod(r - 1, n)
        j = tf.math.mod(r + 1, n)
        adj = tf.SparseTensor(
            tf.concat((tf.stack((r, i), 1), tf.stack((r, j), 1)), 0),
            tf.ones((2 * n,), tf.float32),
            (n, n),
        )
        adj = tf.sparse.reorder(adj)  # pylint: disable=no-value-for-parameter
        # z = transforms.effective_resistance_z(adj)
        z = transforms.approx_effective_resistance_z(adj, jl_factor=100, max_iter=1000)
        i = tf.zeros((n,), dtype=tf.int64)
        actual = transforms.get_effective_resistance(z, i, r)
        expected = (n - r) * r / n
        self.assertAllClose(actual, expected, rtol=0.2)


if __name__ == "__main__":
    tf.test.main()

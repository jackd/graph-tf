import tensorflow as tf

from graph_tf.utils import graph_utils


def edges_to_adj(edges, n):
    edges = tf.convert_to_tensor(edges, tf.int64)
    adj = tf.SparseTensor(edges, tf.ones(edges.shape[0],), (n, n),)
    adj_T = tf.sparse.transpose(adj)
    adj_T = tf.sparse.reorder(adj_T)  # pylint: disable=no-value-for-parameter
    return tf.sparse.add(adj, adj_T)


class GraphUtilsTest(tf.test.TestCase):
    def test_component_labels(self):
        n = 6
        edges = [
            [0, 1],
            [0, 5],
            [3, 4],
        ]
        adj = edges_to_adj(edges, n)
        component_labels = graph_utils.get_component_labels(adj)
        self.assertEqual(component_labels.num_components, 3)
        self.assertAllEqual(component_labels.labels, [0, 0, 1, 2, 2, 0])

    def test_largest_component_indices(self):
        n = 6
        edges = [
            [0, 1],
            [0, 5],
            [3, 4],
        ]
        adj = edges_to_adj(edges, n)
        indices = graph_utils.get_largest_component_indices(adj)
        self.assertAllEqual(indices, [0, 1, 5])

    def test_zero_eigs(self):
        n = 6
        edges = [
            [0, 1],
            [0, 5],
            [3, 4],
        ]
        adj = edges_to_adj(edges, n)
        nc = graph_utils.get_component_labels(adj).num_components
        X = graph_utils.get_laplacian_zero_eigs(adj, symmetric_normalized=False)
        self.assertAllClose(tf.linalg.norm(X, axis=0), tf.ones((nc,)))
        rsqrts = tf.math.rsqrt(tf.range(4, dtype=tf.float32))
        self.assertAllClose(
            X,
            [
                [rsqrts[3], 0, 0],
                [rsqrts[3], 0, 0],
                [0, 1, 0],
                [0, 0, rsqrts[2]],
                [0, 0, rsqrts[2]],
                [rsqrts[3], 0, 0],
            ],
        )
        X = graph_utils.get_laplacian_zero_eigs(
            adj, symmetric_normalized=True, normalize=False
        )
        r2 = tf.sqrt(2.0)
        self.assertAllClose(
            X, [[r2, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0],],
        )

    def test_pairwise_effective_resistance(self):
        i = tf.constant([2, 3, 5])
        Z = tf.random.Generator.from_seed(0).normal((10, 7))

        ii, jj = tf.meshgrid(i, i, indexing="ij")
        naive = graph_utils.get_effective_resistance(Z, ii, jj)

        actual = graph_utils.get_pairwise_effective_resistance(Z, i)
        self.assertAllClose(naive, actual)


if __name__ == "__main__":
    tf.test.main()

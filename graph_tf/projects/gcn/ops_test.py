import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from graph_tf.projects.gcn import ops
from graph_tf.utils.ops import SparseImplementation
from graph_tf.utils.test_utils import random_sparse


def get_multi_graph_data(
    num_adj=3, num_nodes=5, filters_in=7, filters_out=11, nnz=5, seed=0
):
    rng = tf.random.Generator.from_seed(seed)
    adjacencies = [
        random_sparse((num_nodes, num_nodes), nnz + i, rng) for i in range(num_adj)
    ]
    kernel = rng.normal((filters_in, num_adj, filters_out))
    x = rng.normal((num_nodes, filters_in))
    return x, adjacencies, kernel


class GcnOpsTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*SparseImplementation.all())
    def test_graph_conv(self, sparse_impl):
        # compare transform_first
        x, adjacencies, kernel = get_multi_graph_data(num_adj=1)
        (adjacency,) = adjacencies
        kernel = tf.squeeze(kernel, axis=1)

        first = ops.graph_conv(x, adjacency, kernel, sparse_impl, transform_first=True)
        second = ops.graph_conv(
            x, adjacency, kernel, sparse_impl, transform_first=False
        )
        first, second = self.evaluate((first, second))
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)

    @parameterized.parameters(
        ops.multi_graph_conv_v0, ops.multi_graph_conv_v1, ops.multi_graph_conv_v2
    )
    def test_multi_graph_versions_self_consistent(self, fn):
        # assert each version is consistent for all combinations of
        # sparse_impl / transform_first
        args = get_multi_graph_data()
        coo1 = fn(*args, sparse_impl="coo", transform_first=True)

        coo2 = fn(*args, sparse_impl="coo", transform_first=False)
        np.testing.assert_allclose(*self.evaluate((coo1, coo2)), rtol=2e-5)
        csr1 = fn(*args, sparse_impl="csr", transform_first=True)
        np.testing.assert_allclose(*self.evaluate((coo1, csr1)), rtol=2e-5)
        csr2 = fn(*args, sparse_impl="csr", transform_first=False)
        np.testing.assert_allclose(*self.evaluate((coo1, csr2)), rtol=2e-5)

    def test_multi_graph_versions_consistent(self):
        # ensure different versions are consistent
        args = get_multi_graph_data()
        v0 = ops.multi_graph_conv_v0(*args)
        v1 = ops.multi_graph_conv_v1(*args)
        v2 = ops.multi_graph_conv_v2(*args)

        np.testing.assert_allclose(*self.evaluate((v0, v1)), rtol=2e-5)
        np.testing.assert_allclose(*self.evaluate((v0, v2)), rtol=2e-5)


if __name__ == "__main__":
    tf.test.main()
    # GcnOpsTest().test_multi_graph()

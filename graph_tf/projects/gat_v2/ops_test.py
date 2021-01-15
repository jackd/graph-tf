import tensorflow as tf

from graph_tf.projects.gat_v2 import ops
from graph_tf.utils.test_utils import random_sparse


class GatV3OpsTest(tf.test.TestCase):
    def test_consistent(self):
        ni = 5
        no = 7
        nnz = 15
        f = 11
        h = 13
        rng = tf.random.Generator.from_seed(0)

        adj = random_sparse((no, ni), nnz, rng)
        nnz = adj.values.shape[0]
        features = rng.normal((ni, h, f))
        attention = rng.uniform((adj.values.shape[0], h))

        v0 = ops.multi_attention_v0(features, attention, adj)
        v1 = ops.multi_attention_v1(features, attention, adj)

        self.assertAllClose(v0, v1)

        v1_mean = ops.multi_attention_v1(features, attention, adj, reduction="mean")
        self.assertAllClose(tf.reduce_mean(v0, axis=1), v1_mean)
        v1_sum = ops.multi_attention_v1(features, attention, adj, reduction="sum")
        self.assertAllClose(tf.reduce_sum(v0, axis=1), v1_sum)


if __name__ == "__main__":
    tf.test.main()

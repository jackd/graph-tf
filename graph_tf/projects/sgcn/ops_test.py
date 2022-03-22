import tensorflow as tf

from graph_tf.projects.sgcn import ops


class SgcnOpsTest(tf.test.TestCase):
    def test_spectral_convs_consistent(self, seed: int = 0):
        n = 13
        fi = 15
        fo = 17
        e = 7

        rng = tf.random.Generator.from_seed(seed)
        X = rng.normal((n, fi), dtype=tf.float32)
        w = rng.normal((e, fi, fo))
        V = rng.normal((n, e))

        v1 = ops.spectral_graph_conv_v1(X, w, V)
        v2 = ops.spectral_graph_conv_v2(X, w, V)
        self.assertAllClose(v1, v2, rtol=1e-4)

    def test_channelwise_conv_consistent(self, seed: int = 0):
        n = 13
        e = 7
        f = 5
        rng = tf.random.Generator.from_seed(seed)

        X = rng.normal((n, f), dtype=tf.float32)
        w = rng.normal((e,))
        V = rng.normal((n, e))

        v1 = [
            ops.spectral_graph_conv(tf.expand_dims(Xi, 1), tf.reshape(w, (e, 1, 1)), V)
            for Xi in tf.unstack(X, axis=1)
        ]
        v1 = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            v1, axis=1
        )
        cw = ops.channelwise_spectral_graph_conv(X, w, V)
        self.assertAllClose(v1, cw, rtol=1e-4)

    def test_depthwise_conv_consistent(self, seed: int = 0):
        n = 13
        e = 7
        f = 5
        rng = tf.random.Generator.from_seed(seed)

        X = rng.normal((n, f), dtype=tf.float32)
        w = rng.normal((e, f))
        V = rng.normal((n, e))

        v1 = [
            ops.spectral_graph_conv(tf.expand_dims(Xi, 1), tf.reshape(wi, (e, 1, 1)), V)
            for (Xi, wi) in zip(tf.unstack(X, axis=1), tf.unstack(w, axis=1))
        ]
        v1 = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            v1, axis=1
        )
        cw = ops.depthwise_spectral_graph_conv(X, w, V)
        self.assertAllClose(v1, cw, rtol=1e-4)

    def test_separable_conv_consistent(self, seed: int = 0):
        n = 13
        e = 7
        fi = 5
        fo = 17
        d = 3

        rng = tf.random.Generator.from_seed(seed)

        X = rng.normal((n, fi), dtype=tf.float32)
        w = rng.normal((e, fi * d))
        V = rng.normal((n, e))
        pointwise_kernel = rng.normal((fi * d, fo))

        v1 = tf.matmul(ops.depthwise_spectral_graph_conv(X, w, V), pointwise_kernel)
        v2 = ops.separable_spectral_graph_conv(X, w, V, pointwise_kernel)

        self.assertAllClose(v1, v2, rtol=1e-4)


if __name__ == "__main__":
    tf.test.main()

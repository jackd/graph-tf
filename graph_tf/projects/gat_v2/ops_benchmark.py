import tensorflow as tf

import tfbm
from graph_tf.projects.gat_v2 import ops
from graph_tf.utils.test_utils import random_sparse

# pylint: disable=no-self-use


def get_inputs(ni, no, nnz, f, h):
    rng = tf.random.Generator.from_seed(0)

    adj = random_sparse((no, ni), nnz, rng)
    nnz = tf.shape(adj.values)[0]
    features = rng.normal((ni, h, f))
    attention = rng.uniform((nnz, h))
    return features, attention, adj


class GatV3OpsBenchmark(tfbm.Benchmark):
    BENCHMARK_SPEC = [
        tfbm.benchmark(
            device="gpu", args=(19717, 19717, 108365, 8, 8), name="pubmed-concat",
        ),
        tfbm.benchmark(
            device="gpu",
            args=(19717, 19717, 108365, 8, 8),
            kwargs=dict(reduction="mean"),
            name="pubmed-mean",
        ),
    ]

    @tfbm.benchmark
    def v0(self, *args, reduction=None):
        features, attention, adj = get_inputs(*args)
        with tf.GradientTape() as tape:
            tape.watch((features, attention))
            out = ops.multi_attention_v0(features, attention, adj)
            if reduction == "sum":
                out = tf.reduce_sum(out, axis=1)
            elif reduction == "mean":
                out = tf.reduce_mean(out, axis=1)
            else:
                assert reduction is None
            out = tf.reduce_sum(out)
        return tape.gradient(out, (features, attention))

    @tfbm.benchmark
    def v1(self, *args, reduction=None):
        features, attention, adj = get_inputs(*args)
        with tf.GradientTape() as tape:
            tape.watch((features, attention))
            out = ops.multi_attention_v1(features, attention, adj, reduction=reduction)
            out = tf.reduce_sum(out)
        return tape.gradient(out, (features, attention))


if __name__ == "__main__":
    import tfbm.cli  # pylint: disable=ungrouped-imports

    tfbm.cli.main()

"""
Script for benchmarking performance of matmuls with `SparseTensor` vs `CSRSparseMatrix`.

Requires tensorflow 2.3 and absl-py

```bash
pip install tensorflow==2.3
pip install absl-py
```
"""
import tensorflow as tf
from absl import flags

import tfbm.cli
from graph_tf.projects.gcn import ops
from graph_tf.utils.ops import SparseImplementation
from graph_tf.utils.test_utils import random_sparse
from tfbm.benchmarks import Benchmark, benchmark

flags.DEFINE_integer("ni", default=1024 * 16, help="number of nodes in")
flags.DEFINE_integer("no", default=1024 * 16, help="number of nodes out")
flags.DEFINE_integer("nn", default=8, help="mean number of neighbors")
flags.DEFINE_integer("na", default=5, help="number of adjacency matrices")
flags.DEFINE_integer("fi", default=64, help="number of filters in")
flags.DEFINE_integer("fo", default=64, help="number of filters out")
flags.DEFINE_integer("burn", default=10, help="number of burn iterations")
flags.DEFINE_integer("iters", default=100, help="number of iterations to average over")
flags.DEFINE_boolean("jit", default=False, help="XLA jit compilation")
flags.DEFINE_integer("version", default=0, help="version, in [0, 1, 2]")
flags.DEFINE_string(
    "impl",
    default=SparseImplementation.COO,
    help=f"Use sparse implementation to use, one of {SparseImplementation.all()}",
)
flags.DEFINE_integer("seed", default=0, help="seed for random number generation")
flags.DEFINE_boolean("transform_first", default=False, help="do kernel transform first")
FLAGS = flags.FLAGS


def get_conv_args():
    rng = tf.random.Generator.from_seed(FLAGS.seed)
    nnz = FLAGS.nn * FLAGS.ni
    x = rng.normal((FLAGS.ni, FLAGS.fi), dtype=tf.float32)
    adjacencies = [
        random_sparse((FLAGS.no, FLAGS.ni), nnz, rng) for _ in range(FLAGS.na)
    ]
    kernel = rng.normal((FLAGS.fi, FLAGS.na, FLAGS.fo))
    return x, adjacencies, kernel


def do_conv(conv_fn, backwards=False, **kwargs):
    x, adjacencies, kernel = get_conv_args()

    with tf.GradientTape() as tape:
        params = (x, kernel)
        tape.watch(params)
        x = conv_fn(x, adjacencies, kernel, **kwargs)
        grad = tape.gradient(x, params)
    if backwards:
        return grad
    return x


class GCNOpsBenchmark(Benchmark):
    BENCHMARK_SPEC = [
        benchmark(device="gpu", xla_jit=False, name="forwards", args=(False,)),
        benchmark(device="gpu", xla_jit=True, name="forwards", args=(False,)),
        benchmark(device="gpu", xla_jit=False, name="backwards", args=(True,)),
        benchmark(device="gpu", xla_jit=True, name="backwards", args=(True,)),
    ]

    @benchmark
    def v0(self, backwards):
        return do_conv(ops.multi_graph_conv_v0, backwards=backwards)

    @benchmark
    def v1(self, backwards):
        return do_conv(ops.multi_graph_conv_v1, backwards=backwards)

    @benchmark
    def v2(self, backwards):
        return do_conv(ops.multi_graph_conv_v2, backwards=backwards)


if __name__ == "__main__":
    tfbm.cli.main()

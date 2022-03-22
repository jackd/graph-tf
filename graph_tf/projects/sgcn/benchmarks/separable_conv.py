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
from graph_tf.projects.sgcn import ops
from tfbm.benchmarks import Benchmark, benchmark

flags.DEFINE_integer("n", default=1024 * 16, help="number of nodes")
flags.DEFINE_integer("e", default=16, help="eigendecomposition rank")
flags.DEFINE_integer("fi", default=64, help="number of filters in")
flags.DEFINE_integer("fo", default=64, help="number of filters out")
flags.DEFINE_integer("d", default=2, help="number of filters out")
flags.DEFINE_integer("burn", default=10, help="number of burn iterations")
flags.DEFINE_integer("iters", default=100, help="number of iterations to average over")
flags.DEFINE_integer("seed", default=0, help="rng seed")
FLAGS = flags.FLAGS


def get_conv_args():
    rng = tf.random.Generator.from_seed(FLAGS.seed)
    x = rng.normal((FLAGS.n, FLAGS.fi))
    w = rng.normal((FLAGS.e, FLAGS.fi * FLAGS.d))
    v = rng.normal((FLAGS.n, FLAGS.e))
    kernel = rng.normal((FLAGS.fi * FLAGS.d, FLAGS.fo))
    return x, w, v, kernel


def do_conv(conv_fn, backwards=False, **kwargs):
    x, w, v, kernel = get_conv_args()

    with tf.GradientTape() as tape:
        params = (x, w, kernel)
        tape.watch(params)
        x = conv_fn(x, w, v, kernel, **kwargs)
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
    def sequential(self, backwards):
        def sequential_impl(x, w, v, kernel):
            return tf.matmul(ops.depthwise_spectral_graph_conv(x, w, v), kernel)

        return do_conv(sequential_impl, backwards=backwards)

    @benchmark
    def actual(self, backwards):
        return do_conv(ops.separable_spectral_graph_conv, backwards=backwards)


if __name__ == "__main__":
    tfbm.cli.main()

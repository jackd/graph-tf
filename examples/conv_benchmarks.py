import tensorflow as tf
import tensorflow.python.ops.linalg.sparse.sparse as sparse_lib  # pylint: disable=no-name-in-module

from graph_tf.utils import test_utils
from tfbm import Benchmark, benchmark


def get_args(
    num_nodes: int = 1024,
    sparsity: float = 0.01,
    num_convs: int = 8,
    filters_in: int = 63,
    filters_out: int = 65,
    seed: int = 0,
):
    rng = tf.random.Generator.from_seed(seed)
    nnz = int(num_nodes ** 2 * sparsity)
    adj = test_utils.random_sparse((num_nodes, num_nodes), nnz, rng)
    indices = adj.indices
    nnz = tf.shape(indices)[0]
    adj_values = rng.uniform((num_convs, nnz))
    kernel = rng.normal((num_convs, filters_in, filters_out))
    features = rng.normal((num_nodes, filters_in))
    args = adj_values, features, kernel, indices, adj.dense_shape
    params = args[:3]
    return args, params


def forward_or_backward(fn, backward: bool, **kwargs):
    args, params = get_args(**kwargs)
    if backward:
        with tf.GradientTape() as tape:
            tape.watch(params)
            out = fn(*args)
        return tape.gradient(out, params)
    return fn(*args)


def unstacked_sparse_matmul(adj_values, features, kernel, indices, dense_shape):
    convs = []
    for av, k in zip(tf.unstack(adj_values, axis=0), tf.unstack(kernel, axis=0)):
        adj = tf.SparseTensor(indices, av, dense_shape)
        f = tf.sparse.sparse_dense_matmul(adj, features)
        convs.append(tf.matmul(f, k))
    return tf.add_n(convs)


def unstacked_csr_matmul(adj_values, features, kernel, indices, dense_shape):
    convs = []
    for av, k in zip(tf.unstack(adj_values, axis=0), tf.unstack(kernel, axis=0)):
        adj = tf.SparseTensor(indices, av, dense_shape)
        adj = sparse_lib.CSRSparseMatrix(adj)
        f = sparse_lib.matmul(adj, features)
        convs.append(tf.matmul(f, k))
    return tf.add_n(convs)


def gather_einsum(adj_values, features, kernel, indices, dense_shape):
    del dense_shape
    i, j = tf.unstack(indices, axis=-1)
    features = tf.gather(features, j, axis=0)
    values = tf.einsum("ke,ei,kio->eo", adj_values, features, kernel)
    return tf.math.segment_sum(values, i)


def gather_sum(adj_values, features, kernel, indices, dense_shape):
    del dense_shape
    i, j = tf.unstack(indices, axis=-1)
    k, fi, fo = kernel.shape
    features = tf.gather(features, j, axis=0)
    adj_values = tf.transpose(kernel, adj_values)
    values = tf.expand_dims(features, axis=1) * tf.expand_dims(adj_values, axis=-1)
    values = tf.reshape(values, (tf.shape(i)[0], k * fi))
    kernel = tf.reshape(kernel, (k * fi, fo))
    values = tf.matmul(values, kernel)

    return tf.math.segment_sum(values, i)


class ConvBenchmark(Benchmark):
    BENCHMARK_SPEC = [
        benchmark(
            device="gpu", xla_jit=False, name="forward", kwargs=dict(backward=False)
        ),
        benchmark(
            device="gpu", xla_jit=True, name="forward", kwargs=dict(backward=False)
        ),
        benchmark(
            device="gpu", xla_jit=False, name="backward", kwargs=dict(backward=True)
        ),
        benchmark(
            device="gpu", xla_jit=True, name="backward", kwargs=dict(backward=True)
        ),
    ]

    @benchmark
    def unstacked_sparse_matmul(self, backward: bool = False, **kwargs):
        return forward_or_backward(unstacked_sparse_matmul, backward=backward, **kwargs)

    @benchmark
    def unstacked_csr_matmul(self, backward: bool = False, **kwargs):
        return forward_or_backward(unstacked_sparse_matmul, backward=backward, **kwargs)

    @benchmark
    def gather_einsum(self, backward: bool = False, **kwargs):
        return forward_or_backward(gather_einsum, backward=backward, **kwargs)

    @benchmark
    def gather_sum(self, backward: bool = False, **kwargs):
        return forward_or_backward(gather_einsum, backward=backward, **kwargs)

    # def test_consistent(self):
    #     args, _ = get_args()
    #     unstacked = unstacked_sparse_matmul(*args)
    #     gathered = gather_einsum(*args)
    #     self.assertAllClose(unstacked, gathered)


if __name__ == "__main__":
    import tfbm.cli

    tfbm.cli.main()

import tensorflow as tf

from graph_tf.utils.arnoldi import arnoldi_iteration
from graph_tf.utils.test_utils import random_sparse
from tfbm import Benchmark, benchmark


def get_inputs(num_nodes, mean_degree, p, seed=0):
    nnz = num_nodes * mean_degree
    rng = tf.random.Generator.from_seed(seed)
    A = random_sparse((num_nodes, num_nodes), nnz, rng)
    b = rng.normal((num_nodes, p))
    return A, b


class ArnoldiBenchmark(Benchmark):
    BENCHMARK_SPEC = [
        benchmark(device="cpu", args=(4, int(1e6), 16, 16), name="large"),
        benchmark(device="gpu", args=(4, int(1e5), 16, 16), name="small"),
        benchmark(device="cpu", args=(4, int(1e5), 16, 16), name="small"),
        benchmark(device="gpu", args=(4, int(1e5), 16, 16), name="small", xla_jit=True),
        benchmark(device="cpu", args=(4, int(1e5), 16, 16), name="small", xla_jit=True),
    ]

    @benchmark
    def arnoldi(self, n: int, num_nodes: int, mean_degree: int, p: int):
        A, b = get_inputs(num_nodes, p, mean_degree)
        Q, h = arnoldi_iteration(A, b, n)
        del h
        return Q

    @benchmark
    def repeated_mul(self, n: int, num_nodes: int, mean_degree: int, p: int):
        A, b = get_inputs(num_nodes, p, mean_degree)
        Q = [b]
        for _ in range(n):
            Q.append(tf.sparse.sparse_dense_matmul(A, Q[-1]))
        return tf.stack(Q, axis=0)


if __name__ == "__main__":
    from tfbm.cli import main  # pylint: disable=ungrouped-imports

    main()

import numpy as np
import tensorflow as tf

from graph_tf.data import single
from graph_tf.projects.spag.ops import chebyshev_subspace_iteration_sparse
from graph_tf.utils.ops import to_laplacian
from graph_tf.utils.test_utils import random_laplacian
from tfbm import Benchmark, benchmark

# pylint: disable=no-self-use


def get_random_inputs(
    size: int, nnz: int, k: int, dtype: tf.DType = tf.float32, seed: int = 0
):
    rng = tf.random.Generator.from_seed(seed)
    a, _ = random_laplacian(size, nnz, rng, normalize=False, dtype=dtype)
    v0 = rng.normal((size, k), dtype=dtype)
    return dict(data=a.values.numpy(), indices=a.indices.numpy(), v0=v0.numpy())


def get_citations_inputs(
    k: int,
    name: str = "pub_med",
    dtype: tf.DType = tf.float32,
    normalize: bool = True,
    shift: float = -2.0,
    seed: int = 0,
):
    data = single.citations_data(name=name, largest_component_only=True)
    size = data.labels.shape[0]
    laplacian = to_laplacian(
        data.adjacency.with_values(tf.cast(data.adjacency.values, dtype)),
        normalize=normalize,
        shift=shift,
    )
    rng = tf.random.Generator.from_seed(seed)
    v0 = rng.normal((size, k), dtype=dtype)
    l0 = rng.normal((size, k), dtype=dtype)
    return dict(
        data=laplacian.values.numpy(),
        indices=laplacian.indices.numpy(),
        v0=v0.numpy(),
        l0=l0.numpy(),
    )


class EighBenchmark(Benchmark):
    BENCHMARK_SPEC = [
        benchmark(device="cpu", kwargs=get_citations_inputs(k=4)),
        benchmark(device="gpu", kwargs=get_citations_inputs(k=4)),
    ]
    # BENCHMARK_SPEC = [
    #     benchmark(device="gpu", kwargs=get_random_inputs(k=4, size=1000, nnz=10000))
    # ]

    # @benchmark(
    #     kwargs=dict(**get_kwargs(1000, 10000, k=4), sparse_impl="csr", backward=False),
    #     name="small-csr-fwd",
    # )
    # @benchmark(
    #     kwargs=dict(**get_kwargs(1000, 10000, k=4), sparse_impl="csr", backward=True),
    #     name="small-csr-bwd",
    # )
    # @benchmark(kwargs=dict(sparse_impl="coo", backward=True), name="small-coo-bwd")
    @benchmark(kwargs=dict(sparse_impl="coo", backward=False), name="coo-fwd")
    # @benchmark(kwargs=dict(sparse_impl="coo", backward=True), name="coo-rev")
    def chebyshev_subspace_iteration(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        v0: np.ndarray,
        l0: np.ndarray,
        sparse_impl: str,
        backward: bool,
    ):
        data = tf.convert_to_tensor(data)
        indices = tf.convert_to_tensor(indices)
        v0 = tf.convert_to_tensor(v0)
        l0 = tf.convert_to_tensor(l0)
        m, _ = v0.shape
        a = tf.SparseTensor(indices, data, (m, m))
        w, v = chebyshev_subspace_iteration_sparse(a, v0, l0, sparse_impl=sparse_impl)
        if backward:
            return tf.gradients(tf.reduce_sum(w) + tf.reduce_sum(v), data)
        return w, v


if __name__ == "__main__":
    import tfbm.cli

    tfbm.cli.main()

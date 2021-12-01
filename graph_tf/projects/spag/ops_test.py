import tensorflow as tf
from jax.config import config
from jju.linalg import utils

from graph_tf.projects.spag import ops
from graph_tf.utils.test_utils import random_laplacian

# from jju.linalg import subspace_iteration as si
# from jju.sparse import coo
# import jax.numpy as jnp

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


class SpagOpsTest(tf.test.TestCase):
    def test_chebyshev_subspace_iteration_sparse_values(self):
        rng = tf.random.Generator.from_seed(0)
        n = 100
        nnz = 1000
        k = 4
        laplacian, row_sum = random_laplacian(n, nnz, rng, normalize=True, shift=2.0)
        v0 = rng.normal((n, k), dtype=tf.float32, stddev=0.01) + tf.expand_dims(
            row_sum ** 0.5, axis=1
        )
        l0 = rng.normal((n, k), dtype=tf.float32)

        # data = laplacian.values.numpy()
        # row, col = laplacian.indices.numpy().T
        # sized = jnp.zeros((n,), dtype=bool)
        # a = coo.matmul_fun(data, row, col, sized)
        # w_jax, v_jax, info = si.chebyshev_subspace_iteration(
        #     8, 2.0, a, v0.numpy(), max_iters=1000
        # )

        w_actual, v_actual = ops.chebyshev_subspace_iteration_sparse(
            laplacian, v0, l0, scale=2.0, max_iters=1000, order=8
        )
        w_expected, v_expected = tf.linalg.eigh(tf.sparse.to_dense(laplacian))
        print(w_actual)
        print(w_expected)

        w_expected = w_expected[:k]
        v_expected = v_expected[:, :k]
        v_expected = utils.standardize_signs(v_expected)
        self.assertAllClose(w_actual, w_expected)
        self.assertAllClose(v_actual, v_expected)

    # def test_lobpcg_coo_gradient(self):
    #     rng = tf.random.Generator.from_seed(0)
    #     n = 20
    #     nnz = 200
    #     k = 3
    #     dtype = tf.float64
    #     largest = False
    #     laplacian = random_laplacian(n, nnz, rng, dtype=dtype, normalize=True)
    #     v0 = rng.normal((n, k), dtype=dtype)
    #     data = laplacian.values
    #     row, col = tf.unstack(laplacian.indices, axis=1)

    #     def f(data):
    #         w, v = ops.lobpcg_coo(data, row, col, v0, largest=largest, k=k)
    #         return tf.reduce_sum(w) + tf.reduce_sum(v)

    #     ## `tf.test.compute_gradient` hangs...
    #     # grad_analytic, grad_fd = tf.test.compute_gradient(f, (data,))
    #     # (grad_analytic,) = grad_analytic
    #     # (grad_fd,) = grad_fd
    #     # self.assertAllClose(grad_analytic, grad_fd, atol=2e-5)

    #     with tf.GradientTape() as tape:
    #         tape.watch(data)
    #         out = f(data)
    #     grad = tape.gradient(out, data)
    #     delta = rng.normal(data.shape, dtype=data.dtype)
    #     delta /= tf.linalg.norm(delta)
    #     eps = 1e-4
    #     actual = (f(data + eps * delta) - out) / eps
    #     expected = tf.reduce_sum(grad * delta)
    #     self.assertAllClose(actual, expected)

    # def test_lobpcg_csr_gradient(self):
    #     rng = tf.random.Generator.from_seed(0)
    #     n = 20
    #     nnz = 200
    #     k = 3
    #     largest = False
    #     laplacian = random_laplacian(n, nnz, rng, normalize=True)
    #     v0 = rng.normal((n, k), dtype=tf.float32)
    #     data = laplacian.values
    #     row, col = tf.unstack(laplacian.indices, axis=1)
    #     indices = col
    #     row_lengths = tf.math.bincount(tf.cast(row, tf.int32), maxlength=n)
    #     indptr = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    #         (
    #             tf.zeros((1,), dtype=row_lengths.dtype),
    #             tf.math.cumsum(row_lengths, axis=0),
    #         ),
    #         axis=0,
    #     )

    #     def f(data):
    #         w, v = ops.lobpcg_csr(data, indices, indptr, v0, largest=largest, k=k)
    #         # w, v = ops.lobpcg_csr(
    #         #     data, indices=indices, indptr=indptr, v0=v0, largest=largest, k=k
    #         # )
    #         # return tf.reduce_sum(w)
    #         # return tf.reduce_sum(v)
    #         return tf.reduce_sum(w) + tf.reduce_sum(v)

    #     ## `tf.test.compute_gradient` hangs...
    #     # grad_analytic, grad_fd = tf.test.compute_gradient(f, (data,))
    #     # (grad_analytic,) = grad_analytic
    #     # (grad_fd,) = grad_fd
    #     # self.assertAllClose(grad_analytic, grad_fd, atol=2e-5)

    #     with tf.GradientTape() as tape:
    #         tape.watch(data)
    #         out = f(data)
    #     grad = tape.gradient(out, data)
    #     delta = rng.normal(data.shape, dtype=data.dtype)
    #     delta /= tf.linalg.norm(delta)
    #     eps = 1e-6
    #     actual = (f(data + eps * delta) - out) / eps
    #     expected = tf.reduce_sum(grad * delta)
    #     self.assertAllClose(actual, expected)


if __name__ == "__main__":
    # tf.test.main()
    SpagOpsTest().test_chebyshev_subspace_iteration_sparse_values()

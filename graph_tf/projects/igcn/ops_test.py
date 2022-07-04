import typing as tp

import numpy as np
import tensorflow as tf

from graph_tf.projects.igcn import ops
from graph_tf.projects.igcn.scalable import jl


def get_random_laplacian(
    rng: tf.random.Generator, n: int, sparsity: float = 0.1, eps: float = 0
):
    A = rng.uniform((n, n)) / 2
    mask = rng.uniform((n, n)) < sparsity
    A = tf.where(mask, A, tf.zeros_like(A))
    A = A + tf.transpose(A)
    d = tf.reduce_sum(A, 1)
    d = tf.math.rsqrt(d)
    return tf.eye(n) - (1 - eps) * A * d * tf.expand_dims(d, axis=-1)


class OpsTest(tf.test.TestCase):
    def test_sparse_cg_rank1(self):
        n = 50
        sparsity = 0.1
        eps = 0.1
        rng = tf.random.Generator.from_seed(0)
        A = get_random_laplacian(rng, n, sparsity, eps)
        b = rng.normal((n,))

        A_st = tf.sparse.from_dense(A)
        actual = ops.sparse_cg(A_st, b, max_iter=100).x
        expected = tf.squeeze(tf.linalg.solve(A, tf.expand_dims(b, 1)), 1)
        self.assertAllClose(actual, expected, rtol=1e-3)

    def test_sparse_cg_rank2(self):
        n = 50
        k = 5
        sparsity = 0.1
        eps = 0.1
        rng = tf.random.Generator.from_seed(0)
        A = get_random_laplacian(rng, n, sparsity, eps)
        b = rng.normal((n, k))

        A_st = tf.sparse.from_dense(A)
        actual = ops.sparse_cg(A_st, b, max_iter=100).x
        expected = tf.linalg.solve(A, b)
        self.assertAllClose(actual, expected, rtol=2e-3)

    def test_sparse_cg_gradient(self):
        def get_grad(operator, x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                out = operator(x)
                loss = tf.reduce_sum(out**2)
            return tape.gradient(loss, x)

        n = 50
        k = 5
        sparsity = 0.1
        eps = 0.1
        rng = tf.random.Generator.from_seed(0)
        A = get_random_laplacian(rng, n, sparsity, eps)
        b = rng.normal((n, k))

        A_st = tf.sparse.from_dense(A)
        actual = get_grad(lambda b: ops.sparse_cg(A_st, b, max_iter=100).x, b)
        expected = get_grad(lambda b: tf.linalg.solve(A, b), b)
        self.assertAllClose(actual, expected, rtol=2e-3)

    def test_quadratic_sparse_categorical_crossentropy_terms(
        self,
        num_examples: int = 5,
        num_classes: int = 3,
        from_logits: bool = True,
        seed: int = 0,
        epsilon: float = 1e-2,
        rtol=2e-2,
        atol=1e-6,
    ):
        rng = tf.random.Generator.from_seed(seed)
        y_true = rng.uniform((num_examples,), maxval=num_classes, dtype=tf.int64)
        weights = rng.uniform((num_examples,))

        y_pred = epsilon * rng.normal((num_examples, num_classes))
        if not from_logits:
            y_pred = tf.nn.softmax(y_pred)

        def get_terms(loss_fn: tp.Callable):
            # tf.hessians only supported in graph mode
            @tf.function
            def fn(y_pred):
                loss = loss_fn(y_true, y_pred, from_logits=from_logits)
                assert loss.shape == weights.shape, (loss.shape, weights.shape, loss_fn)
                loss = tf.reduce_sum(loss * weights)
                (grad,) = tf.gradients(loss, y_pred)
                H = tf.hessians(loss, y_pred)
                return loss, grad, H

            return fn(y_pred)

        expected_loss, expected_grad, expected_h = get_terms(
            tf.keras.backend.sparse_categorical_crossentropy
        )

        actual_loss, actual_grad, actual_h = get_terms(
            ops.quadratic_sparse_categorical_crossentropy
        )
        actual_loss = actual_loss + tf.reduce_sum(weights) * np.log(num_classes)

        self.assertAllClose(actual_loss, expected_loss, rtol=rtol, atol=atol)
        self.assertAllClose(actual_grad, expected_grad, rtol=rtol, atol=atol)
        self.assertAllClose(actual_h, expected_h, rtol=rtol, atol=atol)

    def test_lazy_quadratic_categorical_crossentropy(
        self,
        num_labels: int = 7,
        batch_size: int = 5,
        num_classes: int = 3,
        seed: int = 0,
    ):
        rng = tf.random.Generator.from_seed(seed)
        y_true = rng.uniform((num_labels,), maxval=num_classes, dtype=tf.int64)
        AT = rng.uniform((batch_size, num_labels))
        ATY = tf.linalg.matmul(AT, tf.one_hot(y_true, num_classes))

        y_pred = rng.normal((batch_size, num_classes))

        expected = tf.reduce_sum(
            ops.quadratic_sparse_categorical_crossentropy(
                y_true, tf.matmul(AT, y_pred, transpose_a=True)
            )
        )
        actual = ops.lazy_quadratic_categorical_crossentropy(ATY, AT, y_pred)
        self.assertAllClose(actual, expected)

    def test_lazy_quadratic_categorical_crossentropy_v2(
        self,
        num_labels: int = 7,
        batch_size: int = 5,
        num_classes: int = 3,
        seed: int = 0,
    ):
        rng = tf.random.Generator.from_seed(seed)
        y_true = rng.uniform((num_labels,), maxval=num_classes, dtype=tf.int64)
        AT = rng.uniform((batch_size, num_labels))
        ATY = tf.linalg.matmul(AT, tf.one_hot(y_true, num_classes))
        ATA = tf.linalg.matmul(AT, AT, transpose_b=True)

        y_pred = rng.normal((batch_size, num_classes))

        expected = tf.reduce_sum(
            ops.quadratic_sparse_categorical_crossentropy(
                y_true, tf.matmul(AT, y_pred, transpose_a=True)
            )
        )
        actual = ops.lazy_quadratic_categorical_crossentropy_v2(ATY, ATA, y_pred)
        self.assertAllClose(actual, expected)

    def test_projected_lazy_quadratic_categorical_crossentropy(
        self,
        num_labels: int = 1000,
        batch_size: int = 5,
        num_classes: int = 3,
        seed: int = 0,
        jl_eps: float = 0.1,
    ):
        rng = tf.random.Generator.from_seed(seed)
        y_true = rng.uniform((num_labels,), maxval=num_classes, dtype=tf.int64)
        k = jl.johnson_lindenstrauss_min_dim(num_labels, eps=jl_eps)
        AT = rng.uniform((batch_size, num_labels))
        ATR = AT @ jl.gaussian_projection_matrix(k, num_labels, transpose=True, rng=rng)
        ATY = tf.linalg.matmul(AT, tf.one_hot(y_true, num_classes))

        y_pred = rng.normal((batch_size, num_classes))

        expected = ops.lazy_quadratic_categorical_crossentropy(ATY, AT, y_pred)
        actual = ops.lazy_quadratic_categorical_crossentropy(ATY, ATR, y_pred)
        self.assertAllClose(actual, expected, rtol=jl_eps)


if __name__ == "__main__":
    tf.test.main()
    # OpsTest().test_quadratic_sparse_categorical_crossentropy_terms()
    # OpsTest().test_lazy_quadratic_categorical_crossentropy()
    # OpsTest().test_projected_lazy_quadratic_categorical_crossentropy()

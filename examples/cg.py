import numpy as np
import tensorflow as tf

from graph_tf.utils.linalg import SparseLinearOperator

n = 5
m = 3
a = tf.random.uniform((n, n), dtype=tf.float32)
mask = tf.random.uniform((n, n), dtype=tf.float32) < 0.2
a = tf.where(mask, a, tf.zeros_like(a)) + n * tf.eye(n)
a = tf.sparse.from_dense(a)

lo = SparseLinearOperator(a, is_self_adjoint=True, is_positive_definite=True)


x = tf.random.normal((n,))
# sol = tf.linalg.experimental.conjugate_gradient(lo, x)
# y = sol.x
# x2 = lo.matvec(x)
# np.testing.assert_allclose(x, x2, atol=1e-4)
# print("Passed")

x = tf.random.normal((m, n))
y = tf.vectorized_map(lambda x: tf.linalg.experimental.conjugate_gradient(lo, x).x, x)
x2 = tf.vectorized_map(lo.matvec, y)
np.testing.assert_allclose(x, x2, atol=1e-4)
print("Passed")

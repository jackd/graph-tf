import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from graph_tf.utils.arnoldi import arnoldi_iteration, lanczos_iteration

n = 32
k = 8
seed = 0

rng = np.random.default_rng(seed)
A = rng.normal(size=(n, n))
d = np.arange(n)
A[d, d] = 0
A = -np.abs(A + A.T)
A[d, d] = -A.sum(axis=-1)
A = tf.cast(A, tf.float32)

rng = tf.random.Generator.from_seed(0)
b = rng.normal((n, 1))

Ql, d, l = lanczos_iteration(A, b, k)

Qa, h = arnoldi_iteration(A, b, k)

l = l[:-1, 0]
d = d[..., 0]
diags = tf.stack(
    (
        tf.pad(l, [[0, 1]]),  # pylint:disable=no-value-for-parameter
        d,
        tf.pad(l, [[1, 0]]),  # pylint:disable=no-value-for-parameter
    ),
    axis=0,
)
td = tf.linalg.LinearOperatorTridiag(diags, is_self_adjoint=True)

vals, vecs = tf.linalg.eigh(td.to_dense())
Q = Ql[:, :-1, 0]
vecs = tf.matmul(Q, vecs)
vals_full, vecs_full = tf.linalg.eigh(A)

diffs = np.zeros((n, k))
for i in range(n):
    for j in range(k):
        vf = vecs_full[:, i]
        v = vecs[:, j]
        a = tf.linalg.norm(vf - v)
        b = tf.linalg.norm(vf + v)
        diffs[i, j] = tf.minimum(a, b).numpy()

vals = vals.numpy()
vals_full = vals_full.numpy()
plt.scatter(np.linspace(0, 1, len(vals_full)), vals_full)
plt.scatter(np.linspace(0, 1, len(vals)), vals, marker="x")
plt.figure()
sns.heatmap(diffs)
plt.show()

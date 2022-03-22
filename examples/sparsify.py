# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
import tensorflow as tf

from graph_tf.data.transforms import laplacian, sparsify

rng = tf.random.Generator.from_seed(0)

n = 512
# sparsity = 0.05


def get_edges(offset):
    return tf.stack((r, tf.math.mod(r + offset, n)), 1)


r = tf.range(n, dtype=tf.int64)
edges = tf.concat(
    (
        get_edges(-3),
        get_edges(-2),
        get_edges(-1),
        get_edges(1),
        get_edges(2),
        get_edges(3),
    ),
    axis=0,
)
m = tf.shape(edges)[0]
adj = tf.SparseTensor(edges, tf.ones((m,)), (n, n))
adj = tf.sparse.reorder(adj)
sparse_adj = sparsify(adj, matrix_conc_const=0.01, max_iter=1000, epsilon=0.2)
sparse_L = tf.sparse.to_dense(laplacian(sparse_adj))
L = tf.sparse.to_dense(laplacian(adj))

sparse_u, sparse_v = tf.linalg.eigh(sparse_L)
u, v = tf.linalg.eigh(L)

sparse_v_vt = tf.linalg.matmul(sparse_v, v, transpose_b=True)

x = tf.random.normal((128, n), stddev=tf.math.rsqrt(tf.cast(n, tf.float32)))
xLx = tf.einsum("ki,ij,kj->k", x, L, x)
sparse_xLx = tf.einsum("ki,ij,kj->k", x, sparse_L, x)
print(tf.stack((xLx, sparse_xLx), 1).numpy())
print((xLx / sparse_xLx).numpy())
print("Done")

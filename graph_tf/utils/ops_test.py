import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from graph_tf.utils import ops, scipy_utils
from graph_tf.utils.test_utils import random_sparse


class UtilsOpsTest(tf.test.TestCase):
    def test_ravel_multi_index(self):
        indices = np.array([[0, 1], [2, 3]])
        dense_shape = [7, 5]

        # axis = 0
        expected = np.ravel_multi_index(indices, dense_shape)
        actual = ops.ravel_multi_index(indices, dense_shape, axis=0)
        np.testing.assert_equal(self.evaluate(actual), expected)

        # axis = 1
        expected = np.ravel_multi_index(indices.T, dense_shape)
        actual = ops.ravel_multi_index(indices, dense_shape, axis=1)
        np.testing.assert_equal(self.evaluate(actual), expected)

    def test_unravel_index(self):
        indices = [15, 23, 13, 1, 0]
        dense_shape = [5, 7]

        # axis = 0
        expected = np.unravel_index(indices, dense_shape)
        actual = ops.unravel_index(indices, dense_shape)
        np.testing.assert_equal(self.evaluate(actual), expected)

        # axis = 1
        actual2 = ops.unravel_index(indices, dense_shape, axis=1)
        actual2 = tf.transpose(actual2, (1, 0))
        np.testing.assert_equal(*self.evaluate((actual, actual2)))

    def test_collect_sparse(self):
        i = tf.sparse.SparseTensor([[0, 0], [0, 0]], [1, 2], [1, 1])
        i = ops.collect_sparse(i)
        indices, values = self.evaluate((i.indices, i.values))
        np.testing.assert_equal(indices, [[0, 0]])
        np.testing.assert_equal(values, [3])

    def test_normalize_sparse(self):
        n = 100
        A = random_sparse((n, n), nnz=500, rng=tf.random.Generator.from_seed(0))
        A = tf.sparse.add(A, tf.sparse.eye(n))
        A = A.with_values(tf.abs(A.values))
        values, indices = self.evaluate((A.values, A.indices))
        A_sp = sp.coo_matrix((values, indices.T))
        tf_impl = ops.normalize_sparse(A)
        sp_impl = scipy_utils.normalize_sparse(A_sp)
        sp_impl = sp_impl.tocsr(copy=False).tocoo(copy=False)  # force reorder

        row, col = tf.unstack(tf_impl.indices, axis=-1)
        values = tf_impl.values
        row, col, values = self.evaluate((row, col, values))
        tf_impl = tf.sparse.reorder(tf_impl)  # pylint: disable=no-value-for-parameter

        np.testing.assert_equal(row, sp_impl.row)
        np.testing.assert_equal(col, sp_impl.col)
        np.testing.assert_allclose(values, sp_impl.data, rtol=1e-5)

    def test_chebyshev_polynomials(self):
        n = 10
        nnz = 20
        k = 2
        rng = tf.random.Generator.from_seed(0)
        A = tf.sparse.add(random_sparse((n, n), nnz, rng), tf.sparse.eye(n))
        A = A.with_values(tf.abs(A.values))

        def unpack(st):
            return st.values, tf.unstack(st.indices, axis=-1)

        data, (i, j) = self.evaluate(unpack(A))
        A_sp = sp.coo_matrix((data, (i, j)), shape=(n, n))

        expected = scipy_utils.chebyshev_polynomials(A_sp, k)
        expected = [exp.tocsr(copy=False).tocoo(copy=False) for exp in expected]
        actual = ops.chebyshev_polynomials(A, k)
        actual = self.evaluate([unpack(st) for st in actual])

        for ((data, (row, col)), exp) in zip(actual, expected):
            np.testing.assert_equal(row, exp.row)
            np.testing.assert_equal(col, exp.col)
            np.testing.assert_allclose(data, exp.data, atol=1e-5)

    def test_indices_to_mask(self):
        expected = [0, 1, 0, 0, 0, 1, 0]
        size = len(expected)
        (indices,) = np.where(expected)
        actual = ops.indices_to_mask(indices, size)
        actual = self.evaluate(actual)
        np.testing.assert_equal(actual, expected)

    def test_segment_softmax(self):
        rng = tf.random.Generator.from_seed(0)
        nrows = 5
        ncols = 3
        next_dim = 1
        dense = rng.normal((nrows, ncols, next_dim))
        lengths = rng.uniform((nrows,), 1, ncols, dtype=tf.int64)
        mask = tf.sequence_mask(lengths, maxlen=ncols)
        where_mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, next_dim),)
        dense = tf.where(where_mask, dense, tf.fill((nrows, ncols, next_dim), -1e9))
        expected = tf.boolean_mask(tf.nn.softmax(dense, axis=1), mask)
        ragged = tf.RaggedTensor.from_tensor(dense, lengths)
        data = ragged.values
        segment_ids = ragged.value_rowids()
        num_segments = ragged.nrows()
        actual = ops.segment_softmax(data, segment_ids)
        np.testing.assert_allclose(*self.evaluate((actual, expected)), rtol=1e-6)

        # unsorted
        perm = tf.random.shuffle(tf.range(tf.shape(data)[0]))
        actual = ops.unsorted_segment_softmax(
            tf.gather(data, perm, axis=0),
            tf.gather(segment_ids, perm, axis=0),
            num_segments,
        )
        np.testing.assert_allclose(
            *self.evaluate((actual, tf.gather(expected, perm, axis=0))), rtol=1e-6
        )

    def test_sparse_gather(self):
        # ensure gather_adjacency_sparse is consistent with dense
        num_nodes = 50
        nnz = 150
        num_dims = 2
        rng = tf.random.Generator.from_seed(0)
        st = random_sparse((num_nodes,) * num_dims, nnz, rng=rng)
        dense = tf.sparse.to_dense(st)
        mask = rng.uniform((num_nodes,)) > 0.6
        indices = tf.boolean_mask(tf.range(num_nodes), mask)
        for axis in range(num_dims):
            actual = ops.sparse_gather(st, indices, axis=axis).st
            expected = tf.gather(dense, indices, axis=axis)
            self.assertAllEqual(tf.sparse.to_dense(actual), expected)

    def test_sparse_gather_again(self):
        num_nodes = 50
        nnz = 150
        num_dims = 2
        nf = 3
        ids = tf.constant([10, 20, 30, 35], dtype=tf.int64)
        rng = tf.random.Generator.from_seed(0)
        st = random_sparse((num_nodes,) * num_dims, nnz, rng=rng)
        features = rng.uniform((num_nodes, nf))

        v0 = tf.gather(tf.sparse.sparse_dense_matmul(st, features), ids, axis=0)
        v1 = tf.sparse.sparse_dense_matmul(ops.sparse_gather(st, ids).st, features)
        self.assertAllClose(v0, v1)

    def test_sparse_gather_all(self):
        # ensure gather_adjacency_sparse is consistent with dense
        num_nodes = 50
        nnz = 150
        num_dims = 2
        rng = tf.random.Generator.from_seed(0)
        st = random_sparse((num_nodes,) * num_dims, nnz, rng=rng)
        mask = rng.uniform((num_nodes,)) > 0.6
        indices = tf.boolean_mask(tf.range(num_nodes), mask)
        actual = ops.sparse_gather_all(st, indices).st
        expected = st
        for axis in range(num_dims):
            expected = ops.sparse_gather(expected, indices, axis=axis).st

        self.assertAllEqual(actual.indices, expected.indices)
        self.assertAllEqual(actual.values, expected.values)
        self.assertAllEqual(actual.dense_shape, expected.dense_shape)

    def test_sparse_boolean_mask(self):
        num_nodes = 50
        nnz = 150
        num_dims = 2
        rng = tf.random.Generator.from_seed(0)
        st = random_sparse((num_nodes,) * num_dims, nnz, rng=rng)
        dense = tf.sparse.to_dense(st)
        mask = rng.uniform((num_nodes,)) > 0.6
        for axis in range(num_dims):
            actual = ops.sparse_boolean_mask(st, mask, axis=axis).st
            expected = tf.boolean_mask(dense, mask, axis=axis)
            self.assertAllEqual(tf.sparse.to_dense(actual), expected)

    def test_sparse_boolean_mask_all(self):
        # ensure gather_adjacency_sparse is consistent with dense
        num_nodes = 50
        nnz = 150
        num_dims = 2
        rng = tf.random.Generator.from_seed(0)
        st = random_sparse((num_nodes,) * num_dims, nnz, rng=rng)
        mask = rng.uniform((num_nodes,)) > 0.6
        actual = ops.sparse_boolean_mask_all(st, mask).st
        expected = st
        for axis in range(num_dims):
            expected = ops.sparse_boolean_mask(expected, mask, axis=axis).st

        self.assertAllEqual(actual.indices, expected.indices)
        self.assertAllEqual(actual.values, expected.values)
        self.assertAllEqual(actual.dense_shape, expected.dense_shape)


if __name__ == "__main__":
    tf.test.main()

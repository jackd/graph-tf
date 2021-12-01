from typing import Optional

import tensorflow as tf


class SparseOperator(tf.linalg.LinearOperator):
    """`tf.linalg.LinearOperator` implementation wrapping `tf.SparseTensor`."""

    def __init__(
        self,
        st: tf.SparseTensor,
        is_non_singular: Optional[bool] = None,
        is_self_adjoint: Optional[bool] = None,
        is_positive_definite: Optional[bool] = None,
        is_square: Optional[bool] = None,
        name=None,
    ):
        self.st = st
        super().__init__(
            st.dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
        )

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        return tf.sparse.sparse_dense_matmul(
            self.st, x, adjoint_a=adjoint, adjoint_b=adjoint_arg
        )

    def _shape(self):
        return self.st.shape

    def _shape_tensor(self):
        return self.st.dense_shape


def masked_outer(indices, x, y):
    """Compute `(x @ y.T)`."""
    raise NotImplementedError("TODO")

import functools
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow.python.ops.linalg.sparse.sparse as sparse_lib  # pylint: disable=no-name-in-module

from graph_tf.utils.type_checks import is_dense_tensor, is_sparse_tensor

# from tensorflow.python.util import dispatch  # pylint: disable=no-name-in-module


CSRSparseMatrix = sparse_lib.CSRSparseMatrix


class MaskedSparseResult(NamedTuple):
    st: tf.SparseTensor
    entries_mask: tf.Tensor


def is_csr_matrix(x):
    return isinstance(x, CSRSparseMatrix)


class SparseImplementation:
    COO = "coo"
    CSR = "csr"

    @classmethod
    def all(cls):
        return SparseImplementation.COO, SparseImplementation.CSR

    @classmethod
    def validate(cls, impl: str):
        options = cls.all()
        if impl not in options:
            raise ValueError(
                f"Invalid {cls.__name__} value. Got {impl}, but must be on of {options}"
            )


def to_sparse_impl(
    sp: Union[tf.SparseTensor, CSRSparseMatrix], impl: SparseImplementation
) -> Union[tf.SparseTensor, CSRSparseMatrix]:
    SparseImplementation.validate(impl)
    if impl == SparseImplementation.COO:
        return to_coo(sp)
    if impl == SparseImplementation.CSR:
        return to_csr(sp)
    raise RuntimeError("SparseImplementation.validate error")


def to_coo(sp: Union[tf.SparseTensor, CSRSparseMatrix]) -> tf.SparseTensor:
    if is_sparse_tensor(sp):
        return sp
    if is_csr_matrix(sp):
        return sp.to_sparse_tensor()
    raise TypeError(f"sp must be a `SparseTensor` or `CSRSparseMatrix`, got {sp}")


def to_csr(st: Union[tf.SparseTensor, CSRSparseMatrix]) -> CSRSparseMatrix:
    if is_sparse_tensor(st):
        return CSRSparseMatrix(st)
    if is_csr_matrix(st):
        return st
    raise TypeError(f"sp must be a `SparseTensor` or `CSRSparseMatrix`, got {st}")


def matmul(
    a: Union[tf.Tensor, tf.SparseTensor, CSRSparseMatrix],
    b: Union[tf.Tensor, CSRSparseMatrix],
):
    if is_dense_tensor(a):
        assert is_dense_tensor(b)
        return tf.matmul(a, b)
    if is_dense_tensor(b):
        return sparse_dense_matmul(a, b)
    if is_csr_matrix(b):
        return sparse_sparse_matmul(a, b)
    raise TypeError(f"b must be a Tensor or CSRSparseMatrix, got {b}")


def sparse_tensor(*args):
    if any(tf.keras.backend.is_keras_tensor(arg) for arg in args):
        return tf.keras.layers.Lambda(lambda a: sparse_tensor(*a))(args)
    return tf.SparseTensor(*args)


def get_dense_shape(st: tf.SparseTensor):
    if tf.keras.backend.is_keras_tensor(st):
        return tf.keras.layers.Lambda(get_dense_shape)(st)
    return st.dense_shape


def sparse_stack(sp_inputs: Sequence[tf.SparseTensor], axis: int = 0):
    sp_inputs = [tf.sparse.expand_dims(x, axis=axis) for x in sp_inputs]
    return tf.sparse.concat(sp_inputs=sp_inputs, axis=axis)


def sparse_dense_matmul(a: Union[tf.SparseTensor, CSRSparseMatrix], b: tf.Tensor):
    if is_sparse_tensor(a):
        return tf.sparse.sparse_dense_matmul(a, b)

    if is_csr_matrix(a):
        return sparse_lib.matmul(a, b)

    raise TypeError(f"a must be SparseTensor or CSRSparseMatrix, got {a}")


def sparse_sparse_matmul(a: CSRSparseMatrix, b: CSRSparseMatrix):
    """Sparse-sparse matrix multiplication using CSRSparseMatrix."""
    assert is_csr_matrix(a)
    assert is_csr_matrix(b)
    return sparse_lib.matmul(a, b)


def _ravel_scalar(dense_shape, axis=0):
    dense_shape = tf.convert_to_tensor(dense_shape)
    assert dense_shape.dtype.is_integer
    dense_shape.shape.assert_has_rank(1)
    scalar = tf.math.cumprod(dense_shape, reverse=True, exclusive=True, axis=0)
    shape = [1] * dense_shape.shape[0]
    shape[axis] = -1
    return tf.reshape(scalar, shape)


def ravel_multi_index(indices, dense_shape, axis=0):
    """Get the index into the ravelled (1D) tensor."""
    scalar = _ravel_scalar(dense_shape, axis=axis)
    return tf.reduce_sum(indices * scalar, axis=axis)


def unravel_index(indices, dense_shape, axis=0):
    """Get the index into the unravelled (ND) tensor."""
    indices = tf.convert_to_tensor(indices)
    indices.shape.assert_has_rank(1)
    scalar = _ravel_scalar(dense_shape, axis=axis)
    dense_shape = tf.reshape(dense_shape, tf.shape(scalar))
    indices = tf.expand_dims(indices, axis=axis)
    return indices // scalar % dense_shape


def unique_ravelled(
    indices: tf.Tensor, dense_shape: tf.Tensor, axis: int = 0
) -> Tuple[tf.Tensor, tf.Tensor]:
    indices = ravel_multi_index(indices, dense_shape, axis=axis)
    indices, idx = tf.unique(indices)
    indices = unravel_index(indices, dense_shape, axis=axis)
    return indices, idx


def collect_sparse(st: tf.SparseTensor):
    """Collect values at duplicate indices of st."""
    dense_shape = tf.convert_to_tensor(st.dense_shape)
    indices = st.indices
    indices_1d = ravel_multi_index(indices, dense_shape, axis=-1)
    indices_1d, segments = tf.unique(indices_1d)
    values = tf.math.segment_sum(st.values, segments)
    indices = unravel_index(indices_1d, dense_shape, axis=-1)
    return sparse_tensor(indices, values, dense_shape)


def normalize_sparse(A: tf.SparseTensor, symmetric: bool = True):
    row_sum = tf.sparse.reduce_sum(A, axis=1)
    tf.debugging.assert_non_negative(row_sum)
    i, j = tf.unstack(A.indices, axis=-1)
    if symmetric:
        d_vals = tf.math.rsqrt(row_sum)
        d_vals = tf.where(row_sum == 0, tf.ones_like(d_vals), d_vals)
        values = A.values * tf.gather(d_vals, i, axis=0) * tf.gather(d_vals, j, axis=0)
    else:
        d_vals = tf.math.reciprocal(row_sum)
        d_vals = tf.where(row_sum == 0, tf.ones_like(d_vals), d_vals)
        values = A.values * tf.gather(d_vals, i, axis=0)
    return A.with_values(values)


def largest_eigs(A: tf.SparseTensor, n: int):
    # pylint: disable=import-outside-toplevel
    import scipy.sparse as sp
    from scipy.sparse.linalg.eigen.arpack import eigsh

    # pylint: enable=import-outside-toplevel

    def eigs_np(indices, values, dense_shape):
        A_sp = sp.coo_matrix((values, indices.T), shape=dense_shape)
        return eigsh(A_sp, n, which="LM")[0]

    out = tf.numpy_function(eigs_np, (A.indices, A.values, A.dense_shape), tf.float32)
    out.set_shape((n,))
    return out


def negative(st: tf.SparseTensor) -> tf.SparseTensor:
    return st.with_values(-st.values)


def subtract(st_a: tf.SparseTensor, st_b: tf.SparseTensor) -> tf.SparseTensor:
    return tf.sparse.add(st_a, negative(st_b))


def chebyshev_polynomials(A: tf.SparseTensor, k: int) -> Sequence[tf.SparseTensor]:
    """
    Calculate Chebyshev polynomials up to order k.

    Args:
        A: input sparse matrix
        k: order of chebyshev polynomial

    Returns:
        k+1 sparse tensors
    """

    A = normalize_sparse(A)
    N = A.dense_shape[0]
    laplacian = subtract(tf.sparse.eye(N), A)
    largest_eigval = tf.squeeze(largest_eigs(laplacian, 1))
    scaled_laplacian = subtract(laplacian * (2.0 / largest_eigval), tf.sparse.eye(N))

    t_k = [tf.sparse.eye(N), scaled_laplacian]
    rescaled_laplacian = CSRSparseMatrix(scaled_laplacian * 2)

    for _ in range(2, k + 1):
        t = subtract(
            to_coo(sparse_lib.matmul(rescaled_laplacian, to_csr(t_k[-1]))), t_k[-2]
        )
        t_k.append(t)

    return t_k


def _sparse_dropout(x: tf.SparseTensor, rate: float, uniform_fn: Callable):
    assert x.dtype.is_floating
    mask = uniform_fn(tf.shape(x.values), dtype=tf.float32) >= rate
    st = tf.sparse.retain(x, mask)
    # gradient issues with automatic broadcasting of sparse tensors
    # https://github.com/tensorflow/tensorflow/issues/46008#issuecomment-751755570
    return st.with_values(st.values / (1 - rate))


def sparse_negate(x: tf.SparseTensor):
    return x.with_values(-x.values)


def sparse_dropout(x: tf.SparseTensor, rate: float, seed=None):
    return _sparse_dropout(x, rate, functools.partial(tf.random.uniform, seed=seed))


def sparse_dropout_rng(
    x: tf.SparseTensor, rate: float, rng: Optional[tf.random.Generator] = None
):
    if rng is None:
        rng = tf.random.get_global_generator()
    return _sparse_dropout(x, rate, rng.uniform)


def unsorted_segment_softmax(data: tf.Tensor, segment_ids: tf.Tensor, num_segments):
    # stabilize
    max_vals = tf.gather(
        tf.math.unsorted_segment_max(data, segment_ids, num_segments),
        segment_ids,
        axis=0,
    )
    data = data - max_vals
    # standard softmax
    data = tf.exp(data)
    summed_vals = tf.gather(
        tf.math.unsorted_segment_sum(data, segment_ids, num_segments),
        segment_ids,
        axis=0,
    )
    return data / summed_vals


def segment_softmax(data: tf.Tensor, segment_ids: tf.Tensor):
    # stabilize
    max_vals = tf.gather(tf.math.segment_max(data, segment_ids), segment_ids, axis=0)
    data = data - max_vals
    # standard softmax
    data = tf.exp(data)
    summed_vals = tf.gather(tf.math.segment_sum(data, segment_ids), segment_ids, axis=0)
    return data / summed_vals


def smart_cond(cond, if_true: Callable, if_false: Callable):
    if isinstance(cond, (tf.Tensor, tf.Variable)):
        assert cond.dtype.is_bool
        return tf.cond(cond, if_true, if_false)
    if cond:
        return if_true()
    return if_false()


def finalize_dense(
    x: tf.Tensor,
    bias: Optional[tf.Tensor] = None,
    activation: Optional[Callable] = None,
):
    if bias is not None:
        x = x + bias
    if activation:
        x = activation(x)
    return x


def sparse_dense(
    x: Union[tf.SparseTensor, CSRSparseMatrix],
    kernel: tf.Tensor,
    bias: Optional[tf.Tensor] = None,
    activation: Optional[Callable] = None,
) -> tf.Tensor:
    x = matmul(x, kernel)
    return finalize_dense(x, bias=bias, activation=activation)


def scatter_1d(indices: tf.Tensor, values: tf.Tensor, size: tf.Tensor) -> tf.Tensor:
    indices = tf.convert_to_tensor(indices)
    values = tf.convert_to_tensor(values)
    size = tf.convert_to_tensor(size)
    indices.shape.assert_has_rank(1)
    size.shape.assert_has_rank(0)
    return tf.scatter_nd(tf.expand_dims(indices, axis=1), values, (size,))


def indices_to_mask(indices: tf.Tensor, size: tf.Tensor, dtype=tf.bool) -> tf.Tensor:
    return scatter_1d(indices, tf.ones((tf.size(indices),), dtype=dtype), size)


def _prepare_gather_mask_args(
    indices: Optional[tf.Tensor], mask: Optional[tf.Tensor], old_size: tf.Tensor
):
    if indices is None:
        indices = tf.squeeze(tf.where(mask), axis=1)

    new_size = tf.size(indices, out_type=tf.int64)
    if mask is None:
        mask = scatter_1d(indices, tf.ones((new_size,), tf.bool), old_size)
    indices.shape.assert_has_rank(1)
    mask.shape.assert_has_rank(1)

    return indices, mask, new_size


def _sparse_gather_mask(
    st: tf.SparseTensor,
    *,
    indices: Optional[tf.Tensor] = None,
    mask: Optional[tf.Tensor] = None,
    axis: int = 0,
) -> MaskedSparseResult:
    dense_shape = tf.unstack(get_dense_shape(st))
    old_size = dense_shape[axis]
    indices, mask, new_size = _prepare_gather_mask_args(indices, mask, old_size)
    dense_shape[axis] = new_size
    dense_shape = tf.stack(dense_shape, 0)

    sparse_indices = st.indices
    entries_mask = tf.gather(mask, sparse_indices[:, axis], axis=0)
    sparse_indices = tf.boolean_mask(sparse_indices, entries_mask)
    index_map = scatter_1d(indices, tf.range(new_size), old_size)
    sparse_indices = tf.unstack(sparse_indices, axis=-1)
    sparse_indices[axis] = tf.gather(index_map, sparse_indices[axis], axis=0)
    sparse_indices = tf.stack(sparse_indices, axis=-1)

    values = tf.boolean_mask(st.values, entries_mask)
    st = sparse_tensor(sparse_indices, values, dense_shape)
    return MaskedSparseResult(st, entries_mask)


def sparse_gather(
    st: tf.SparseTensor, indices: tf.Tensor, axis: int = 0
) -> MaskedSparseResult:
    """
    `tf.gather` equivalent for `tf.SparseTensor` values.

    Also returns the mask for the surviving entries.
    """
    return _sparse_gather_mask(st, indices=indices, axis=axis)


def sparse_boolean_mask(
    st: tf.SparseTensor, mask: tf.Tensor, axis: int = 0
) -> MaskedSparseResult:
    """
    `tf.boolean_mask` equivalent for `tf.SparseTensor` values.

    Also returns themask for surviving entries.
    """
    return _sparse_gather_mask(st, mask=mask, axis=axis)


def _sparse_gather_mask_all(
    st: tf.SparseTensor,
    *,
    indices: Optional[tf.Tensor] = None,
    mask: Optional[tf.Tensor] = None,
) -> MaskedSparseResult:
    old_size = st.dense_shape[0]
    indices, mask, new_size = _prepare_gather_mask_args(indices, mask, old_size)

    sparse_indices = st.indices
    entries_mask = tf.reduce_all(tf.gather(mask, sparse_indices, axis=0), axis=-1)
    sparse_indices = tf.boolean_mask(st.indices, entries_mask)

    index_map = scatter_1d(indices, tf.range(new_size), old_size)
    sparse_indices = tf.gather(index_map, sparse_indices, axis=0)

    values = tf.boolean_mask(st.values, entries_mask)
    dense_shape = tf.fill((st.shape.ndims,), new_size)
    st = sparse_tensor(sparse_indices, values, dense_shape)
    return MaskedSparseResult(st, entries_mask)


def sparse_gather_all(st: tf.SparseTensor, indices: tf.Tensor) -> MaskedSparseResult:
    """
    Gather on all axes simultaneously.

    This is equivalent to:
    ```python
    for axis in range(st.shape.ndims):
        st = sparse_gather(st, indices, axis=axis)
    ```

    Args:
        st: `SparseTensor` to gather from.
        indices: rank-1 integer tensor of indices to gather. Entries must all be less
            than the size of each dimension of `st`.

    Returns:
        `SparseTensor` of same rank as `st` and size in each dimension equal to the size
            of `indices`. If `indices` and `st.indices` are sorted then this will have
            sorted indices as well.
        entries_mask: mask of surviving entries, i.e.
            `st.values[entries_mask] == returned_st.values`.
    """
    return _sparse_gather_mask_all(st, indices=indices)


def sparse_boolean_mask_all(st: tf.SparseTensor, mask: tf.Tensor):
    """
    Boolean mask all axes simultaneously.

    This is equivalent to:
    ```python
    for axis in range(st.shape.ndims):
        st = boolean_mask_sparse(st, mask, axis=axis)
    ```

    Args:
        st: square (or hypercube) `SparseTensor` to gather from.
        mask: rank-1 bool mask tensor. Must have the same size as each of the dimensions
            of `st`.

    Returns:
        `SparseTensor` of same rank as `st` and size in each dimension equal to the
            number of True values in `mask`. If `st` has sorted indices then this will
            have sorted indices as well.
        entries_mask: mask of surviving entries, i.e.
            `st.values[entries_mask] == returned_st.values`.
    """
    return _sparse_gather_mask_all(st, mask=mask)


# @dispatch.add_dispatch_support
def krylov(A: Union[tf.Tensor, tf.SparseTensor], x: tf.Tensor, dims: int, axis=1):
    """
    Get vectors of the rank-(dims+1) krylov subspace.

    i.e. [x, A @ x, A**2 @ x, A ** 3 @ x, ...]

    Args:
        A: [n, n] square tensor / SparseTensor.
        x: [n, m] tensor of features.
        dims: number of matrix-matrix multiplications.
        axis: axis to stack outputs on.

    Returns:
        Tensor with the same shape as x except with an additional `dims + 1` axis at
          `axis`.
    """
    out = [x]
    for _ in range(dims):
        x = matmul(A, x)
        out.append(x)
    return tf.stack(out, axis=axis)


def assert_adjacency(adjacency):
    tf.debugging.assert_none_equal(*tf.unstack(adjacency.indices, axis=-1))
    tf.debugging.assert_equal(*tf.unstack(adjacency.dense_shape))


def to_laplacian(
    adjacency: tf.SparseTensor, normalize: bool = False, shift: float = 0.0
):
    weights = adjacency.values
    indices = adjacency.indices
    dense_shape = adjacency.dense_shape
    dtype = adjacency.dtype
    assert dtype.is_floating

    num_nodes = dense_shape[0]

    i, j = tf.unstack(indices, axis=-1)
    diag_values = tf.math.segment_sum(weights, i)
    if normalize:
        factor = tf.math.pow(diag_values, -0.5)
        weights = weights * tf.gather(factor, i, axis=0) * tf.gather(factor, j, axis=0)
        diag_values = tf.ones((num_nodes,), dtype=dtype)
    if shift:
        diag_values = diag_values - shift

    diag = tf.SparseTensor(
        tf.tile(tf.expand_dims(tf.range(num_nodes, dtype=tf.int64), 1), (1, 2)),
        diag_values,
        dense_shape,
    )
    neg_adj = tf.SparseTensor(indices, -weights, dense_shape)
    laplacian = tf.sparse.add(diag, neg_adj)
    return laplacian

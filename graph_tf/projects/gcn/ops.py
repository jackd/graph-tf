from typing import Optional, Sequence, Union

import tensorflow as tf

from graph_tf.utils.ops import (
    SparseImplementation,
    matmul,
    sparse_stack,
    to_sparse_impl,
)
from graph_tf.utils.type_checks import is_dense_tensor


def _transform_first(transform_first, kernel_shape):
    if transform_first is None:
        return kernel_shape[0] >= kernel_shape[-1]
    return transform_first


def graph_conv(
    x: Union[tf.Tensor, tf.SparseTensor],
    adjacency: tf.SparseTensor,
    kernel: Union[tf.Tensor, tf.Variable],
    sparse_impl: SparseImplementation = SparseImplementation.COO,
    transform_first: Optional[bool] = None,
):
    SparseImplementation.validate(sparse_impl)
    kernel = tf.convert_to_tensor(kernel)
    transform_first = _transform_first(transform_first, kernel.shape)
    if not is_dense_tensor(x):
        x = to_sparse_impl(x, sparse_impl)

    adjacency = to_sparse_impl(adjacency, sparse_impl)
    if transform_first:
        return matmul(adjacency, matmul(x, kernel))
    return matmul(matmul(adjacency, x), kernel)


def _validate_shapes(x, adjacencies, kernel):
    filters_in, num_adj, _ = kernel.shape
    assert x.shape[1] == filters_in
    assert len(adjacencies) == num_adj


def multi_graph_conv_v0(
    x: Union[tf.Tensor, tf.SparseTensor],
    adjacencies: Sequence[tf.SparseTensor],
    kernel: Union[tf.Tensor, tf.Variable],
    sparse_impl: SparseImplementation = SparseImplementation.COO,
    transform_first: Optional[bool] = None,
):
    """Implementation based on splitting kernel."""
    SparseImplementation.validate(sparse_impl)
    _validate_shapes(x, adjacencies, kernel)
    kernel = tf.convert_to_tensor(kernel)
    if not is_dense_tensor(x):
        x = to_sparse_impl(x, sparse_impl)

    kernels = tf.unstack(kernel, axis=1)
    return tf.add_n(
        [
            graph_conv(x, adj, k, sparse_impl, transform_first=transform_first)
            for adj, k in zip(adjacencies, kernels)
        ]
    )


def multi_graph_conv_v1(
    x: Union[tf.Tensor, tf.SparseTensor],
    adjacencies: Sequence[tf.SparseTensor],
    kernel: Union[tf.Tensor, tf.Variable],
    sparse_impl: SparseImplementation = SparseImplementation.COO,
    transform_first: Optional[bool] = None,
):
    SparseImplementation.validate(sparse_impl)
    _validate_shapes(x, adjacencies, kernel)
    kernel = tf.convert_to_tensor(kernel)
    transform_first = _transform_first(transform_first, kernel.shape)

    if not is_dense_tensor(x):
        x = to_sparse_impl(x, sparse_impl)

    filters_in, num_adj, filters_out = kernel.shape
    if transform_first:
        kernel = tf.reshape(kernel, (filters_in, num_adj * filters_out))
        x = matmul(x, kernel)
        x = tf.reshape(x, (-1, num_adj, filters_out))
        xs = [
            matmul(to_sparse_impl(adj, sparse_impl), x)
            for adj, x in zip(adjacencies, tf.unstack(x, axis=1))
        ]
        return tf.add_n(xs)

    # transform second
    xs = [matmul(to_sparse_impl(adj, sparse_impl), x) for adj in adjacencies]
    x = tf.reshape(tf.stack(xs, axis=-1), (-1, filters_in * num_adj))
    kernel = tf.reshape(kernel, (filters_in * num_adj, filters_out))
    return matmul(x, kernel)


def multi_graph_conv_v2(
    x: Union[tf.Tensor, tf.SparseTensor],
    adjacencies: Sequence[tf.SparseTensor],
    kernel: Union[tf.Tensor, tf.Variable],
    sparse_impl: SparseImplementation = SparseImplementation.COO,
    transform_first: Optional[bool] = None,
):
    SparseImplementation.validate(sparse_impl)
    _validate_shapes(x, adjacencies, kernel)
    kernel = tf.convert_to_tensor(kernel)
    transform_first = _transform_first(transform_first, kernel.shape)
    nodes_out, nodes_in = tf.unstack(adjacencies[0].dense_shape)
    filters_in, num_adj, filters_out = kernel.shape

    if not is_dense_tensor(x):
        x = to_sparse_impl(x, sparse_impl)

    if transform_first:
        kernel = tf.reshape(kernel, (filters_in, num_adj * filters_out))
        x = matmul(x, kernel)
        x = tf.reshape(x, (nodes_in * num_adj, filters_out))
        adjacency = tf.sparse.reshape(  # pylint: disable=no-value-for-parameter
            sparse_stack(adjacencies, axis=-1), (nodes_out, nodes_in * num_adj)
        )
        return matmul(adjacency, x)

    # transform second
    adjacency = tf.sparse.concat(sp_inputs=adjacencies, axis=1)  # no, a*ni
    adjacency = tf.sparse.reshape(  # pylint: disable=no-value-for-parameter
        adjacency, (nodes_out * num_adj, nodes_in)
    )
    adjacency = to_sparse_impl(adjacency, sparse_impl)
    x = matmul(adjacency, x)
    x = tf.reshape(x, (nodes_out, num_adj * filters_in))
    kernel = tf.transpose(kernel, (1, 0, 2))  # this transpose annoys me greatly
    kernel = tf.reshape(kernel, (num_adj * filters_in, filters_out))
    return tf.matmul(x, kernel)

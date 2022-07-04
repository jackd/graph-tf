import functools
import typing as tp

import gin
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf
import tflo.matrix.dispatch  # pylint: disable=unused-import
from tflo.matrix.core import CompositionMatrix, FullMatrix, Matrix
from tflo.matrix.extras import (
    CGSolverMatrix,
    GatherMatrix,
    ScatterMatrix,
    SparseMatrix,
    StaticPowerSeriesMatrix,
)

from graph_tf.data.single import DataSplit, SemiSupervisedSingle

register = functools.partial(gin.register, module="gtf.igcn.data")


def eliminate_zeros(st: tf.SparseTensor, epsilon: tp.Optional[float] = None):
    """Reorder and remove zero values in `st`."""
    if epsilon is None:
        mask = st.values != 0
    else:
        assert epsilon >= 0
        mask = tf.abs(st.values) <= epsilon
    st = tf.SparseTensor(
        tf.boolean_mask(st.indices, mask),
        tf.boolean_mask(st.values, mask),
        st.dense_shape,
    )
    return st


def _get_cg_solver(st: tf.SparseTensor, tol: float, max_iter: int) -> Matrix:
    sm = SparseMatrix(st, is_self_adjoint=True, is_positive_definite=True)
    return CGSolverMatrix(sm, tol=tol, max_iter=max_iter)


@register
def sparse_cg_solver(
    st: tf.SparseTensor,
    row_ids: tp.Optional[tf.Tensor] = None,
    tol: float = 1e-5,
    max_iter: int = 20,
    preprocess: bool = False,
) -> tp.Union[Matrix, tf.Tensor]:
    solver = _get_cg_solver(st, tol=tol, max_iter=max_iter)
    if row_ids is None:
        if preprocess:
            return solver.to_dense()
        return solver
    gather = GatherMatrix(row_ids, st.shape[0])
    if preprocess:
        return gather.to_dense() @ solver
    comp = CompositionMatrix((gather, solver))
    return comp


@register
def preprocessed_sparse_cg_solver(
    st: tf.SparseTensor,
    row_ids: tp.Optional[tf.Tensor] = None,
    tol: float = 1e-5,
    max_iter: int = 20,
) -> tf.Tensor:
    return sparse_cg_solver(st, row_ids, tol=tol, max_iter=max_iter, preprocess=True)


@register
def low_rank_inverse_approx(
    st: tf.SparseTensor,
    row_ids: tf.Tensor,
    column_ids: tp.Optional[tf.Tensor] = None,
    k: int = 100,
    which: str = "LM",
    shift: float = -2.0,
) -> Matrix:
    indices = st.indices.numpy().T
    values = st.values.numpy()
    coo = sp.coo_matrix((values, indices), shape=st.shape)
    coo = coo + sp.eye(coo.shape[0], dtype=coo.dtype) * shift
    w, v = la.eigsh(coo, k=k, which=which)
    w -= shift
    w = tf.convert_to_tensor(w)
    v = tf.convert_to_tensor(v)
    v_rows = tf.gather(v, row_ids, axis=0)
    V0 = FullMatrix(v_rows / w)
    if column_ids is None:
        VT = FullMatrix(v).adjoint()
    else:
        VT = FullMatrix(tf.gather(v, column_ids, axis=0)).adjoint()
    return CompositionMatrix((V0, VT))


@register
def power_series_inverse(
    st: tf.SparseTensor,
    row_ids: tp.Optional[tf.Tensor] = None,
    column_ids: tp.Optional[tf.Tensor] = None,
    num_terms: int = 10,
) -> Matrix:
    r"""
    Create a power series approximation of the inverse of `st`.

    st^{-1} \approx \sum_{0}^{num_terms} (I - st)^i

    Args:
        st: SparseTensor to approximate inverse.
        row_ids: if not None, gathers these rows after computing the power series.
        num_terms: number of terms in power series.

    Returns:
        Matrix approximating the inverse of `st`.
    """
    st = tf.sparse.add(tf.sparse.eye(st.shape[0]), st.with_values(-st.values))
    st = eliminate_zeros(tf.sparse.reorder(st))
    matrix = uniform_power_series(st, num_terms)
    if row_ids is None and column_ids is None:
        return matrix
    matrices = [matrix]
    if column_ids is not None:
        matrices.append(ScatterMatrix(column_ids, matrix.shape[1]))
    if row_ids is not None:
        matrices.insert(0, GatherMatrix(row_ids, matrix.shape[0]))
    return CompositionMatrix(tuple(matrices))


@register
def uniform_power_series(st: tf.SparseTensor, num_terms: int = 10) -> Matrix:
    r"""Create a power series `\sum_{i=0}^{num_terms} st^i`."""
    return StaticPowerSeriesMatrix(SparseMatrix(st), [1] * num_terms)


@register
def get_deflated_laplacian(A: tf.SparseTensor) -> Matrix:
    """
    Get the non-singular deflated symmetrically-normalized Laplacian.

    `L_deflated = L_sym_normalized - X @ X.T`

    Columns of `X` are the set of eigenvectors with corresponding eigenvalue of 0.

    Args:
        A: sparse adjacency matrix

    Returns:
        L_deflated.
    """
    raise Exception("TODO - requires SumMatrix to tflo")


@register
def get_shifted_laplacian(A: tf.SparseTensor, epsilon: float = 0.1) -> tf.SparseTensor:
    # pylint: disable=no-value-for-parameter
    A = get_discounted_normalized_adjacency(A, epsilon, symmetric=True)
    A = A.with_values(-A.values)
    L = tf.sparse.add(tf.sparse.eye(A.shape[0]), A)
    L = tf.sparse.reorder(L)
    # pylint: enable=no-value-for-parameter
    return L


@register
def get_discounted_normalized_adjacency(
    A: tf.SparseTensor, epsilon: float = 0.1, symmetric: bool = True
) -> tf.SparseTensor:
    d = tf.sparse.reduce_sum(A, axis=1)
    i, j = tf.unstack(A.indices, axis=1)
    if symmetric:
        d = tf.math.rsqrt(d)
        A = A.with_values(
            (1 - epsilon) * A.values * tf.gather(d, i, axis=0) * tf.gather(d, j, axis=0)
        )
    else:
        d = tf.math.reciprocal(d)
        A = A.with_values((1 - epsilon) * A.values * tf.gather(d, i, axis=0))
    return A


Propagator = tp.Union[tf.Tensor, Matrix]
PropagatorFn = tp.Callable[[tf.SparseTensor, tf.Tensor], Propagator]


@register
def get_logit_propagated_split(
    data: SemiSupervisedSingle,
    laplacian_fn: tp.Callable[[tf.SparseTensor], tf.SparseTensor],
    train_propagator_fn: PropagatorFn,
    validation_propagator_fn: tp.Optional[PropagatorFn] = None,
    test_propagator_fn: tp.Optional[PropagatorFn] = None,
) -> DataSplit:
    if validation_propagator_fn is None:
        validation_propagator_fn = train_propagator_fn
    if test_propagator_fn is None:
        test_propagator_fn = validation_propagator_fn
    features = data.node_features
    laplacian = laplacian_fn(data.adjacency)

    def get_data(fn, ids):
        if ids is None:
            return None
        return tf.data.Dataset.from_tensors(
            (
                (fn(laplacian, ids), features),
                tf.gather(data.labels, ids, axis=0),
                tf.fill(ids.shape, 1 / ids.shape[0]),
            )
        )

    return DataSplit(
        get_data(train_propagator_fn, data.train_ids),
        get_data(validation_propagator_fn, data.validation_ids),
        get_data(test_propagator_fn, data.test_ids),
    )


def get_set_complement(ids: tf.Tensor, size: int) -> tf.Tensor:
    mask = tf.scatter_nd(
        tf.expand_dims(ids, axis=1),
        tf.ones(ids.shape, dtype=tf.bool),
        (size,),
    )
    mask = tf.logical_not(mask)
    return tf.squeeze(tf.where(mask), axis=1)


# @register
# def get_batched_logit_propagated_split_v3(
#     data: SemiSupervisedSingle,
#     laplacian_fn: tp.Callable[[tf.SparseTensor], tf.SparseTensor],
#     batch_size: int,
#     tol: float = 1e-5,
#     max_iter: int = 20,
#     preprocess_validation: bool = True,
#     preprocess_test: bool = False,
#     temperature: float = 1.0,  # high temperature -> uniform sampling
#     seed: int = 0,
# ) -> DataSplit:
#     cg_kwargs = dict(tol=tol, max_iter=max_iter)
#     laplacian = laplacian_fn(data.adjacency)
#     train_smoother = sparse_cg_solver(
#         laplacian, data.train_ids, preprocess=True, **cg_kwargs
#     )
#     logits = tf.reduce_sum(train_smoother, axis=0) / temperature

#     train_labels = tf.gather(data.labels, data.train_ids, axis=0)
#     train_weights = tf.fill(data.train_ids.shape, 1 / data.train_ids.shape[0])

#     def map_fn(seed: tf.Tensor):
#         ids = sample_without_replacement(seed, logits, batch_size)
#         features = tf.gather(data.node_features, ids, axis=0)
#         coeff = tf.gather(train_smoother, ids, axis=1)
#         inputs = (coeff, features)
#         return inputs, train_labels, train_weights

#     train_dataset = tf.data.Dataset.random(seed=seed).batch(2).map(map_fn).prefetch(-1)

#     def get_data(ids: tp.Optional[tf.Tensor], preprocess: bool):
#         if ids is None:
#             return None
#         inv = sparse_cg_solver(laplacian, ids, preprocess=preprocess, **cg_kwargs)
#         return tf.data.Dataset.from_tensors(
#             (
#                 (inv, data.node_features),
#                 tf.gather(data.labels, ids, axis=0),
#                 tf.fill(ids.shape, 1 / ids.shape[0]),
#             )
#         )

#     return DataSplit(
#         train_dataset,
#         get_data(data.validation_ids, preprocess_validation),
#         get_data(data.test_ids, preprocess_test),
#     )


# @register
# def get_batched_logit_propagated_split_v2(
#     data: SemiSupervisedSingle,
#     laplacian_fn: tp.Callable[[tf.SparseTensor], tf.SparseTensor],
#     feature_nodes_per_step: int,
#     label_nodes_per_step: int,
#     drop_remainder: bool = True,
#     tol: float = 1e-5,
#     max_iter: int = 20,
#     preprocess_validation: bool = True,
#     preprocess_test: bool = False,
# ) -> DataSplit:
#     """
#     Sample feature_nodes_per_step non-train nodes and label_nodes_per_step train nodes.
#     """
#     tf.debugging.assert_equal(
#         data.train_ids, tf.range(data.train_ids.shape[0], dtype=data.train_ids.dtype)
#     )
#     cg_kwargs = dict(tol=tol, max_iter=max_iter)
#     laplacian = laplacian_fn(data.adjacency)
#     num_nodes = laplacian.shape[1]
#     train_smoother = sparse_cg_solver(
#         laplacian, data.train_ids, preprocess=True, **cg_kwargs
#     )
#     train_ids = data.train_ids
#     train_ids_ds = (
#         tf.data.Dataset.from_tensor_slices(train_ids)
#         .shuffle(train_ids.shape[0])
#         .batch(label_nodes_per_step, drop_remainder=drop_remainder)
#     )

#     not_train_ids = get_set_complement(train_ids, num_nodes)
#     not_train_ids_ds = (
#         tf.data.Dataset.from_tensor_slices(not_train_ids)
#         .shuffle(not_train_ids.shape[0])
#         .repeat()
#         .batch(feature_nodes_per_step)
#     )

#     train_ds = tf.data.Dataset.zip((train_ids_ds, not_train_ids_ds))
#     train_weights = tf.fill(label_nodes_per_step, 1 / data.train_ids.shape[0])

#     def map_fn(train_ids, not_train_ids):
#         ids = (
#             tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
#                 (train_ids, not_train_ids), axis=0
#             )
#         )
#         features = tf.gather(data.node_features, ids, axis=0)
#         coeff = tf.gather(train_smoother, train_ids, axis=0)
#         coeff = tf.gather(coeff, ids, axis=1)
#         inputs = (coeff, features)
#         labels = tf.gather(data.labels, train_ids)
#         return inputs, labels, train_weights

#     train_ds = train_ds.map(map_fn).prefetch(-1)

#     def get_data(ids: tp.Optional[tf.Tensor], preprocess: bool):
#         if ids is None:
#             return None
#         inv = sparse_cg_solver(laplacian, ids, preprocess=preprocess, **cg_kwargs)
#         return tf.data.Dataset.from_tensors(
#             (
#                 (inv, data.node_features),
#                 tf.gather(data.labels, ids, axis=0),
#                 tf.fill(ids.shape, 1 / ids.shape[0]),
#             )
#         )

#     return DataSplit(
#         train_ds,
#         get_data(data.validation_ids, preprocess_validation),
#         get_data(data.test_ids, preprocess_test),
#     )


# @register
# def get_batched_logit_propagated_split(
#     data: SemiSupervisedSingle,
#     laplacian_fn: tp.Callable[[tf.SparseTensor], tf.SparseTensor],
#     batch_size: int,
#     drop_remainder: bool = True,
#     tol: float = 1e-5,
#     max_iter: int = 20,
#     preprocess_validation: bool = True,
#     preprocess_test: bool = False,
#     always_include_train_ids: bool = True,
# ) -> DataSplit:
#     cg_kwargs = dict(tol=tol, max_iter=max_iter)
#     laplacian = laplacian_fn(data.adjacency)
#     num_nodes = laplacian.shape[1]
#     train_smoother = sparse_cg_solver(
#         laplacian, data.train_ids, preprocess=True, **cg_kwargs
#     )

#     train_labels = tf.gather(data.labels, data.train_ids, axis=0)
#     train_weights = tf.fill(data.train_ids.shape, 1 / data.train_ids.shape[0])

#     def map_fn(ids):
#         if always_include_train_ids:
#             ids = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
#                 (ids, data.train_ids), axis=0
#             )
#         ids = tf.sort(ids)
#         features = tf.gather(data.node_features, ids, axis=0)
#         coeff = tf.gather(train_smoother, ids, axis=1)
#         inputs = (coeff, features)
#         return inputs, train_labels, train_weights

#     if always_include_train_ids:
#         train_dataset = tf.data.Dataset.from_tensor_slices(
#             get_set_complement(data.train_ids, num_nodes)
#         )
#     else:
#         train_dataset = tf.data.Dataset.range(num_nodes)
#     train_dataset = (
#         train_dataset.shuffle(len(train_dataset))
#         .batch(batch_size, drop_remainder=drop_remainder)
#         .map(map_fn)
#     )

#     def get_data(ids: tp.Optional[tf.Tensor], preprocess: bool):
#         if ids is None:
#             return None
#         inv = sparse_cg_solver(laplacian, ids, preprocess=preprocess, **cg_kwargs)
#         return tf.data.Dataset.from_tensors(
#             (
#                 (inv, data.node_features),
#                 tf.gather(data.labels, ids, axis=0),
#                 tf.fill(ids.shape, 1 / ids.shape[0]),
#             )
#         )

#     return DataSplit(
#         train_dataset,
#         get_data(data.validation_ids, preprocess_validation),
#         get_data(data.test_ids, preprocess_test),
#     )


@register
def get_input_propagated_split(
    data: SemiSupervisedSingle,
    laplacian_fn: tp.Callable[[tf.SparseTensor], tf.SparseTensor],
    inverse_fn: tp.Callable[[tf.SparseTensor, tf.Tensor], tf.Tensor],
    concat_original: bool = False,
) -> DataSplit:
    features = data.node_features
    laplacian = laplacian_fn(data.adjacency)
    smoothed = inverse_fn(laplacian, features)

    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    features = tf.concat((features, smoothed), axis=1) if concat_original else smoothed

    def get_data(ids):
        if ids is None:
            return None
        return tf.data.Dataset.from_tensors(
            (
                tf.gather(features, ids, axis=0),
                tf.gather(data.labels, ids, axis=0),
                tf.fill((ids.shape[0],), 1 / ids.shape[0]),
            )
        )

    # pylint: enable=unexpected-keyword-arg,no-value-for-parameter

    DataSplit(
        get_data(data.train_ids),
        get_data(data.validation_ids),
        get_data(data.test_ids),
    )

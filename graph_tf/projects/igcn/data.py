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
    ProgMatrix,
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


def _get_cg_solver(
    st: tf.SparseTensor, tol: float, max_iter: int, show_progress: bool = False
) -> Matrix:
    sm = SparseMatrix(st, is_self_adjoint=True, is_positive_definite=True)
    solver = CGSolverMatrix(sm, tol=tol, max_iter=max_iter)
    if show_progress:
        solver = ProgMatrix(solver)
    return solver


@register
def sparse_cg_solver(
    st: tf.SparseTensor,
    row_ids: tp.Optional[tf.Tensor] = None,
    tol: float = 1e-5,
    max_iter: int = 20,
    preprocess: bool = False,
    show_progress: bool = False,
) -> tp.Union[Matrix, tf.Tensor]:
    solver = _get_cg_solver(st, tol=tol, max_iter=max_iter, show_progress=show_progress)
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

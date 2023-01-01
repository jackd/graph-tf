import functools
import typing as tp

import gin
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import stfu.dispatch  # pylint: disable=unused-import
import tensorflow as tf
import tflo.matrix.dispatch  # pylint: disable=unused-import
import tqdm
from tflo.matrix.core import (
    CompositionMatrix,
    DiagMatrix,
    FullMatrix,
    Matrix,
    ScaledIdentityMatrix,
    composition_matrix,
)
from tflo.matrix.extras import (
    CGSolverMatrix,
    GatherMatrix,
    ProgMatrix,
    ScatterMatrix,
    SparseMatrix,
    StaticPowerSeriesMatrix,
    SumMatrix,
)

from graph_tf.data import transforms
from graph_tf.data.single import DataSplit, SemiSupervisedSingle
from graph_tf.utils import graph_utils, scipy_utils

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
    st: tp.Union[tf.SparseTensor, Matrix],
    tol: float,
    max_iter: int,
    show_progress: bool = False,
) -> Matrix:
    if isinstance(st, tf.SparseTensor):
        sm = SparseMatrix(st, is_self_adjoint=True, is_positive_definite=True)
    elif isinstance(st, Matrix):
        assert st.is_self_adjoint
        assert st.is_positive_definite
        sm = st
    else:
        raise TypeError(f"`st` must be SparseTensor or Matrix, got {type(st)}")
    solver = CGSolverMatrix(
        sm,
        tol=tol,
        max_iter=max_iter,
        x0=tf.ones((st.shape[0],), dtype=st.dtype) / np.sqrt(st.shape[0]),
    )
    if show_progress:
        solver = ProgMatrix(solver)
    return solver


@register
def sparse_cg_solver(
    A: tp.Union[tf.SparseTensor, Matrix],
    row_ids: tp.Optional[tf.Tensor] = None,
    *,
    epsilon: float = 0.1,
    tol: float = 1e-5,
    max_iter: int = 20,
    symmetric: bool = True,
    renormalized: bool = False,
    rescaled: bool = False,
    preprocess: bool = False,
    show_progress: bool = False,
) -> tp.Union[Matrix, tf.Tensor]:
    I = tf.sparse.eye(A.shape[0], dtype=A.dtype)
    if renormalized:
        A = tf.sparse.add(A, I)
    if symmetric:
        L, _ = get_normalized_shifted_laplacian(A, epsilon=epsilon)
        mat = _get_cg_solver(L, tol=tol, max_iter=max_iter, show_progress=show_progress)
    else:
        L, d = get_shifted_laplacian(A, epsilon=epsilon)
        mat = composition_matrix(
            _get_cg_solver(L, tol=tol, max_iter=max_iter, show_progress=show_progress),
            DiagMatrix(d, is_self_adjoint=True),
        )

    if row_ids is not None:
        row_matrix = GatherMatrix(row_ids, A.shape[0])
        if preprocess:
            mat = FullMatrix(row_matrix.to_dense() @ mat)
        else:
            mat = CompositionMatrix((row_matrix, mat))
    if rescaled:
        mat = ScaledIdentityMatrix(mat.shape[0], epsilon) @ mat

    def _to_dense(mat: Matrix):

        if isinstance(mat, CompositionMatrix):
            # LinearOperatorComposition._to_dense implementation is terrible
            return mat.operators[0] @ _to_dense(composition_matrix(*mat.operators[1:]))
        return mat.to_dense()

    if preprocess:
        return _to_dense(mat)
    return mat


@register
def low_rank_inverse_approx(
    A: tf.SparseTensor,
    row_ids: tp.Optional[tf.Tensor] = None,
    column_ids: tp.Optional[tf.Tensor] = None,
    epsilon: float = 0.1,
    k: int = 100,
    which: str = "LM",
) -> Matrix:
    st, _ = get_normalized_shifted_laplacian(A, epsilon=0)
    indices = st.indices.numpy().T
    values = st.values.numpy()
    coo = sp.coo_matrix((values, indices), shape=st.shape)
    coo = coo - sp.eye(coo.shape[0], dtype=coo.dtype) * 2
    w, v = la.eigsh(coo, k=k, which=which)
    w += epsilon + 2
    w = tf.convert_to_tensor(w)
    v = tf.convert_to_tensor(v)
    v_rows = v if row_ids is None else tf.gather(v, row_ids, axis=0)
    V0 = FullMatrix(v_rows / w)
    if column_ids is None:
        VT = FullMatrix(v).adjoint()
    else:
        VT = FullMatrix(tf.gather(v, column_ids, axis=0)).adjoint()
    return CompositionMatrix((V0, VT))


@register
def power_series_shifted_laplacian_inverse(
    A: tf.SparseTensor,
    row_ids: tp.Optional[tf.Tensor] = None,
    epsilon: float = 0.1,
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
    st, _ = get_normalized_adjacency(A)
    st = st * (1 - epsilon)
    matrix = StaticPowerSeriesMatrix(SparseMatrix(st), [1] * num_terms)
    if row_ids is None and column_ids is None:
        return matrix
    matrices = [matrix]
    if column_ids is not None:
        matrices.append(ScatterMatrix(column_ids, matrix.shape[1]))
    if row_ids is not None:
        matrices.insert(0, GatherMatrix(row_ids, matrix.shape[0]))
    return composition_matrix(tuple(matrices))


# @register
# def get_pseudo_inverse_propagators_v3(
#     A: tf.SparseTensor,
#     row_ids: tf.Tensor,
#     *,
#     rank: tp.Optional[int] = None,
# ):
#     num_components, _ = graph_utils.get_component_labels(A, directed=False)
#     L = transforms.normalized_laplacian(A, symmetric=True)
#     mat = (
#       scipy_utils.to_scipy(L) - 2 * sp.eye(L.shape[0], dtype=L.dtype.as_numpy_dtype)
#     )
#     if rank is None:
#         w, v = np.linalg.eigh(mat.todense())
#     else:
#         w, v = la.eigsh(mat, k=rank)
#     w += 2
#     eigengap = w[num_components]
#     w[:num_components] = eigengap
#     w[:num_components] *= 10  # HACK
#     v = tf.convert_to_tensor(v, dtype=tf.float32)
#     w = tf.convert_to_tensor(w, tf.float32)
#     return (tf.gather(v, row_ids) / w) @ tf.transpose(v)


# @register
# def get_pseudo_inverse_propagators_v2(
#     A: tf.SparseTensor,
#     row_ids: tf.Tensor,
#     *,
#     include_uut: bool = True,
#     preprocess_uut: bool = True,
#     single_propagator: bool = False,
#     rank: int = 500,
# ):
#     num_nodes = A.shape[0]
#     d = tf.sparse.reduce_sum(A, axis=1, keepdims=False)
#     num_components, labels = graph_utils.get_component_labels(A, directed=False)
#     u0 = tf.scatter_nd(
#         tf.stack((tf.range(num_nodes, dtype=labels.dtype), labels), axis=1),
#         tf.sqrt(d),
#         shape=(num_nodes, num_components),
#     )
#     u0 = u0 / tf.linalg.norm(u0, axis=0, keepdims=True)
#     L = transforms.normalized_laplacian(A, symmetric=True)

#     deflated = scipy_utils.DeflatedLinearOperator(
#         scipy_utils.to_scipy(L) - 2 * sp.eye(num_nodes, dtype=L.dtype.as_numpy_dtype),
#         u0.numpy(),
#         -2.0,
#     )
#     w, v = la.eigsh(deflated, k=rank)
#     w += 2
#     eigengap = w[0]

#     base = tf.gather(v, row_ids) / w @ tf.transpose(v)
#     if include_uut:
#         u0_gathered = tf.gather(u0, row_ids) / eigengap
#         if preprocess_uut:
#             uut = u0_gathered @ tf.transpose(u0)
#         else:
#             uut = FullMatrix(u0_gathered) @ FullMatrix(u0).adjoint()
#         assert uut.shape == base.shape, (uut.shape, base.shape)
#         if single_propagator:
#             if tf.is_tensor(uut) and tf.is_tensor(base):
#                 return base + uut

#             if tf.is_tensor(uut):
#                 uut = FullMatrix(uut)
#             if tf.is_tensor(base):
#                 base = FullMatrix(base)
#             return uut + base
#         return (base, uut)
#     return base


@register
def get_pseudo_inverse_propagators_disjoint_merge(
    A: tf.SparseTensor,
    row_ids: tf.Tensor,
    *,
    tol: float = 1e-5,
    max_iter: int = 20,
    diag_always_one: bool = True,
    add_propagators: bool = False,
):
    num_nodes = A.shape[0]
    num_components, labels = graph_utils.get_component_labels(
        A, directed=False, dtype=tf.int32
    )
    edge_labels = tf.gather(labels, A.indices[:, 0])
    split_edge_indices = tf.dynamic_partition(A.indices, edge_labels, num_components)
    split_values = tf.dynamic_partition(A.values, edge_labels, num_components)
    component_sizes = tf.unstack(
        tf.math.unsorted_segment_sum(
            tf.ones((labels.shape[0],), tf.int64), labels, num_components
        )
    )
    split_node_indices = tf.dynamic_partition(
        tf.range(num_nodes, dtype=tf.int64), labels, num_components
    )

    row_mask = tf.scatter_nd(
        tf.expand_dims(row_ids, axis=1),
        tf.ones(row_ids.shape, dtype=bool),
        (num_nodes,),
    )
    row_mask_indices = tf.scatter_nd(
        tf.expand_dims(row_ids, axis=1),
        row_ids,
        (num_nodes,),
    )
    split_row_masks = tf.dynamic_partition(row_mask, labels, num_components)
    split_row_mask_indices = tf.dynamic_partition(
        row_mask_indices, labels, num_components
    )
    base = np.zeros((row_ids.shape[0], num_nodes), dtype=np.float32)
    uut = np.zeros_like(base)
    for (
        edge_indices,
        values,
        size,
        row_mask,
        row_mask_indices,
        node_indices,
    ) in tqdm.tqdm(
        zip(
            split_edge_indices,
            split_values,
            component_sizes,
            split_row_masks,
            split_row_mask_indices,
            split_node_indices,
        ),
        total=num_components,
        desc="Computing disjoint pinv",
    ):
        row_ids = tf.squeeze(tf.where(row_mask), axis=1)
        inverse_node_indices = tf.scatter_nd(
            tf.expand_dims(node_indices, axis=1), tf.range(size), (num_nodes,)
        )
        edge_indices = tf.gather(inverse_node_indices, edge_indices)
        if row_ids.shape[0] == 0:
            # no labels
            continue
        base_, uut_ = get_pseudo_inverse_propagators(
            A=tf.SparseTensor(edge_indices, values, (size, size)),
            row_ids=row_ids,
            preprocess=True,
            include_uut=True,
            preprocess_uut=True,
            add_propagators=False,
            diag_always_one=diag_always_one,
            max_iter=max_iter,
            tol=tol,
        )
        row_indices = tf.boolean_mask(row_mask_indices, row_mask)
        inverse_row_indices = tf.scatter_nd(
            tf.expand_dims(row_indices, axis=1),
            tf.range(row_ids.shape[0], dtype=tf.int64),
            (num_nodes,),
        )
        row_indices = tf.gather(inverse_row_indices, row_indices)
        row_indices = row_indices.numpy()
        base[row_indices][:, node_indices] += base_
        uut[row_indices][:, node_indices] += uut_

    if add_propagators:
        return tf.convert_to_tensor(base, uut)
    return tf.convert_to_tensor(base), tf.convert_to_tensor(uut)


def _get_eigen_props(A: tf.SparseTensor, rank: int, *, diag_always_one: bool = False):
    num_nodes = A.shape[0]
    d = tf.sparse.reduce_sum(A, axis=1, keepdims=False)
    u0_vals = tf.sqrt(d)
    if not diag_always_one:
        u0_vals = tf.where(d == 0, tf.ones_like(u0_vals, dtype=A.dtype), u0_vals)
    num_components, labels = graph_utils.get_component_labels(A, directed=False)
    u0 = tf.scatter_nd(
        tf.stack((tf.range(num_nodes, dtype=labels.dtype), labels), axis=1),
        u0_vals,
        shape=(num_nodes, num_components),
    )
    tf.debugging.assert_all_finite(u0, "u0 not all finite")
    norm = tf.linalg.norm(u0, axis=0, keepdims=True)
    tf.debugging.assert_positive(norm, "norms must all be positive")
    u0 = u0 / norm
    # u0 = u0 / tf.where(norm == 0, tf.ones_like(norm), norm)
    # tf.debugging.assert_all_finite(u0, "normalized u0 not all finite")
    L = transforms.normalized_laplacian(
        A, symmetric=True, diag_always_one=diag_always_one
    )
    np.testing.assert_allclose(
        tf.sparse.sparse_dense_matmul(L, u0).numpy(), np.zeros(u0.shape), atol=1e-5
    )

    shifted_L = scipy_utils.to_scipy(L) - 2 * sp.eye(
        num_nodes, dtype=L.dtype.as_numpy_dtype
    )

    deflated = scipy_utils.ShiftedLinearOperator(
        shifted_L,
        u0.numpy(),
        2.0,
    )
    w, v = la.eigsh(
        deflated,
        v0=np.ones((num_nodes,), dtype=deflated.dtype) / np.sqrt(num_nodes),
        k=min(rank, num_nodes - num_components),
    )
    w += 2
    w = tf.convert_to_tensor(w, dtype=L.dtype)
    v = tf.convert_to_tensor(v, dtype=L.dtype)
    return L, u0, w, v


@register
def get_heat_propagator(
    A: tf.SparseTensor,
    row_ids: tf.Tensor,
    *,
    t: float = 1.0,
    preprocess: bool = True,
    renormalized: bool = False,
) -> tp.Union[Matrix, tf.Tensor]:
    mat = transforms.heat_matrix(
        A, t=t, symmetric=True, renormalized=renormalized, preprocess=False
    )
    rows = GatherMatrix(row_ids, mat.shape[0])
    if preprocess:
        return rows.to_dense() @ mat
    return rows @ mat


@register
def get_pseudo_inverse_propagators(
    A: tf.SparseTensor,
    row_ids: tf.Tensor,
    *,
    # tol: float = 1e-5,
    # max_iter: int = 20,
    preprocess: bool = True,
    # show_progress: bool = False,
    # preprocess_uut: bool = True,
    include_uut: bool = True,
    add_propagators: bool = False,
    diag_always_one: bool = False,
    rank: int = 200,
):
    L, u0, w, v = _get_eigen_props(A, rank=rank, diag_always_one=diag_always_one)
    eigengap = w[0]
    tf.debugging.assert_positive(eigengap, "eigengap must be positive")
    tf.debugging.assert_all_finite(w, "eigenvalues must be finite")

    del L

    def outer(u, v):
        if preprocess:
            return tf.linalg.matmul(u, v, adjoint_b=True)
        return FullMatrix(u) @ FullMatrix(v).adjoint()

    base = outer(tf.gather(v, row_ids) * (eigengap / w), v)
    if include_uut:
        uut = outer(tf.gather(u0, row_ids), u0)
        if add_propagators:
            return base + uut
        return base, uut
    return base

    # u0m = FullMatrix(u0)
    # u0m_adjoint = u0m.adjoint()
    # uut = u0m @ u0m_adjoint
    # projector = ScaledIdentityMatrix(num_nodes, tf.ones((), dtype=A.dtype)) - uut
    # # is_positive_definite should actually be False, but we strip nullspace
    # L = SparseMatrix(L, is_self_adjoint=True, is_positive_definite=True)
    # # L = CompositionMatrix(
    # #     (L, projector),
    # #     is_self_adjoint=True,
    # #     is_positive_definite=True,
    # # )

    # mat = CGSolverMatrix(
    #     L,
    #     max_iter=max_iter,
    #     tol=tol,
    # )
    # # mat = CompositionMatrix((mat, projector))
    # if show_progress:
    #     mat = ProgMatrix(mat)
    # rows = GatherMatrix(row_ids, num_nodes)
    # base = rows.to_dense() @ mat if preprocess else CompositionMatrix((rows, mat))

    # if tf.is_tensor(base):
    #     tf.debugging.assert_all_finite(base, "pinv values must be finite")
    #     base = eigengap * base
    #     base = FullMatrix(base)
    # else:
    #     base = CompositionMatrix(
    #         (ScaledIdentityMatrix(row_ids.shape[0], eigengap), base)
    #     )
    # base = CompositionMatrix(((base, projector)))

    # if include_uut:
    #     u0_gathered = tf.gather(u0, row_ids) / eigengap
    #     if preprocess_uut:
    #         uut = u0_gathered @ tf.transpose(u0)
    #     else:
    #         uut = FullMatrix(u0_gathered) @ u0m_adjoint
    #     assert uut.shape == base.shape, (uut.shape, base.shape)
    #     if add_propagators:
    #         if tf.is_tensor(uut) and tf.is_tensor(base):
    #             return base + uut

    #         if tf.is_tensor(uut):
    #             uut = FullMatrix(uut)
    #         if tf.is_tensor(base):
    #             base = FullMatrix(base)
    #         return uut + base
    #     return (base, uut)
    # return base


# @register
# def get_deflated_laplacian(A: tf.SparseTensor) -> Matrix:
#     """
#     Get the non-singular deflated symmetrically-normalized Laplacian.

#     `L_deflated = L_sym_normalized @ (I - U @ U.adjoint())`

#     Columns of `U` are the set of eigenvectors with corresponding eigenvalue of 0.

#     Args:
#         A: sparse adjacency matrix

#     Returns:
#         L_deflated.
#     """
#     num_nodes = A.shape[0]
#     d = tf.sparse.reduce_sum(A, axis=1, keepdims=False)
#     num_components, labels = graph_utils.get_component_labels(A)
#     values = tf.gather(tf.where(d == 0, tf.ones_like(d), tf.sqrt(d)), labels)
#     u0 = tf.scatter_nd(
#         tf.stack((tf.range(num_nodes, dtype=labels.dtype), labels), axis=1),
#         values,
#         shape=(num_nodes, num_components),
#     )
#     u0 = u0 / tf.linalg.norm(u0, axis=0, keepdims=True)
#     L = transforms.normalized_laplacian(A, symmetric=True)

#     L = SparseMatrix(L)
#     u0 = FullMatrix(u0)
#     I = ScaledIdentityMatrix(L.shape[0], tf.ones((), dtype=L.dtype))
#     return L @ (I - u0 @ u0.adjoint())


def add_identity(st: tf.SparseTensor) -> tf.SparseTensor:
    return tf.sparse.add(st, tf.sparse.eye(st.shape[0], dtype=st.dtype))


def get_shifted_laplacian(
    A: tf.SparseTensor, epsilon: float = 0.1
) -> tp.Tuple[tf.SparseTensor, tf.Tensor]:
    d = tf.sparse.reduce_sum(A, axis=1)
    D = tf.sparse.eye(A.shape[0], dtype=A.dtype).with_values(d)
    L = tf.sparse.add(D, A.with_values(A.values * (epsilon - 1)))
    return L, d


def get_normalized_shifted_laplacian(
    A: tf.SparseTensor,
    epsilon: float = 0.1,
    symmetric: bool = True,
) -> tp.Tuple[tf.SparseTensor, tf.Tensor]:
    A, d = get_normalized_adjacency(A, symmetric=symmetric)
    L = add_identity(A.with_values(A.values * (epsilon - 1)))
    L = tf.sparse.reorder(L)
    return L, d


def get_normalized_adjacency(
    A: tf.SparseTensor, symmetric: bool = True
) -> tp.Tuple[tf.SparseTensor, tf.Tensor]:
    d = tf.sparse.reduce_sum(A, axis=1)
    i, j = tf.unstack(A.indices, axis=1)
    if symmetric:
        # d_rsqrt = tf.where(d == 0, tf.zeros_like(d), tf.math.rsqrt(d))
        d_rsqrt = tf.math.rsqrt(d)
        A = A.with_values(
            A.values * tf.gather(d_rsqrt, i, axis=0) * tf.gather(d_rsqrt, j, axis=0)
        )
    else:
        A = A.with_values(A.values * tf.gather(tf.math.reciprocal(d), i, axis=0))
    tf.debugging.assert_all_finite(A.values, "normalized values not finite")
    return A, d


Propagator = tp.Union[tf.Tensor, Matrix]
PropagatorFn = tp.Callable[[tf.SparseTensor, tf.Tensor], Propagator]


@register
def get_logit_propagated_split(
    data: SemiSupervisedSingle,
    train_propagator_fn: PropagatorFn,
    validation_propagator_fn: tp.Optional[PropagatorFn] = None,
    test_propagator_fn: tp.Optional[PropagatorFn] = None,
) -> DataSplit:
    if validation_propagator_fn is None:
        validation_propagator_fn = train_propagator_fn
    if test_propagator_fn is None:
        test_propagator_fn = validation_propagator_fn
    features = data.node_features
    adj = data.adjacency

    def get_data(fn, ids):
        if ids is None:
            return None
        return tf.data.Dataset.from_tensors(
            (
                (fn(adj, ids), features),
                tf.gather(data.labels, ids, axis=0),
                tf.fill(ids.shape, 1 / ids.shape[0]),
            )
        )

    return DataSplit(
        get_data(train_propagator_fn, data.train_ids),
        get_data(validation_propagator_fn, data.validation_ids),
        get_data(test_propagator_fn, data.test_ids),
    )

    # def get_set_complement(ids: tf.Tensor, size: int) -> tf.Tensor:
    #     mask = tf.scatter_nd(
    #         tf.expand_dims(ids, axis=1),
    #         tf.ones(ids.shape, dtype=tf.bool),
    #         (size,),
    #     )
    #     mask = tf.logical_not(mask)
    #     return tf.squeeze(tf.where(mask), axis=1)

    # @register
    # def get_input_propagated_split(
    #     data: SemiSupervisedSingle,
    #     adjacency_transform: tp.Callable[[tf.SparseTensor], tf.SparseTensor],
    #     inverse_fn: tp.Callable[[tf.SparseTensor, tf.Tensor], tf.Tensor],
    #     concat_original: bool = False,
    # ) -> DataSplit:
    #     features = data.node_features
    #     laplacian = adjacency_transform(data.adjacency)
    #     smoothed = inverse_fn(laplacian, features)

    #     # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    # features = (
    #     tf.concat((features, smoothed), axis=1) if concat_original else smoothed
    # )


#     def get_data(ids):
#         if ids is None:
#             return None
#         return tf.data.Dataset.from_tensors(
#             (
#                 tf.gather(features, ids, axis=0),
#                 tf.gather(data.labels, ids, axis=0),
#                 tf.fill((ids.shape[0],), 1 / ids.shape[0]),
#             )
#         )

#     # pylint: enable=unexpected-keyword-arg,no-value-for-parameter

#     DataSplit(
#         get_data(data.train_ids),
#         get_data(data.validation_ids),
#         get_data(data.test_ids),
#     )

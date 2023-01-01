import functools
import typing as tp

import gin
import numpy as np
import scipy.sparse.linalg as la
import tensorflow as tf
import tqdm
from tflo.extras import LinearOperatorCGSolver, LinearOperatorSparseMatrix
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
    ExponentialMatrix,
    MappedMatrix,
    ProgMatrix,
    SparseMatrix,
)

from graph_tf.utils import scipy_utils
from graph_tf.utils.graph_utils import (
    approx_effective_resistance_z,
    laplacian,
    signed_incidence,
    tril,
    tril_indices,
)
from graph_tf.utils.ops import normalize_sparse
from graph_tf.utils.random_utils import as_rng
from graph_tf.utils.type_checks import is_sparse_tensor

register = functools.partial(gin.register, module="gtf.data.transforms")
configurable = functools.partial(gin.configurable, module="gtf.data.transforms")

# back compatible support
register(laplacian)
register(signed_incidence)
register(tril)


@register
def transformed(base, transforms: tp.Union[tp.Callable, tp.Iterable[tp.Callable]]):
    if transforms is None:
        return base
    if callable(transforms):
        return transforms(base)
    for transform in transforms:
        base = transform(base)
    return base


@register
def add_identity(x: tf.SparseTensor) -> tf.SparseTensor:
    if isinstance(x, tf.SparseTensor):
        return tf.sparse.add(x, tf.sparse.eye(x.shape[0], dtype=x.dtype))
    assert tf.is_tensor(x)
    return x + tf.eye(tf.shape(x)[0])


@register
def normalize_symmetric(x: tp.Union[tf.Tensor, tf.SparseTensor]) -> tf.SparseTensor:
    if is_sparse_tensor(x):
        return normalize_sparse(x, symmetric=True)
    d = tf.reduce_sum(tf.abs(x), axis=1)
    d = tf.where(d <= 0, tf.zeros_like(d), tf.math.rsqrt(d))
    return x * d * tf.expand_dims(d, -1)


@register
def normalize_asymmetric(x: tp.Union[tf.Tensor, tf.SparseTensor]) -> tf.SparseTensor:
    if is_sparse_tensor(x):
        return normalize_sparse(x, symmetric=False)
    D = tf.reduce_sum(x, axis=1, keepdims=True)
    D = tf.where(D == 0, tf.zeros_like(D), tf.math.reciprocal(D))
    return x * D


@register
def sparsify(
    adj: tf.SparseTensor,
    *,
    epsilon: float = 0.3,
    matrix_conc_const: float = 4.0,
    jl_factor: float = 4.0,
    rng: tp.Union[int, tf.random.Generator] = 0,
    **cg_kwargs,
) -> tf.SparseTensor:
    rng: tf.random.Generator = as_rng(rng)
    Z = approx_effective_resistance_z(adj, jl_factor=jl_factor, rng=rng, **cg_kwargs)
    i, j, v = tril_indices(adj, return_values=True)
    m = tf.shape(i, tf.int64)[0]

    n = tf.cast(adj.dense_shape[0], tf.float32)

    Zi = tf.gather(Z, i, axis=0)
    Zj = tf.gather(Z, j, axis=0)
    prs = tf.reduce_sum(tf.math.squared_difference(Zi, Zj), axis=1)
    prs = prs * matrix_conc_const * tf.math.log(n) / (epsilon**2) * v
    prs = tf.minimum(prs, tf.ones_like(prs))
    print(prs)
    mask = rng.uniform((m,)) < v * prs
    i, j, v, pr = (tf.boolean_mask(vec, mask) for vec in (i, j, v, prs))
    indices = tf.stack((i, j), axis=1)
    tril = tf.SparseTensor(indices, v / pr, adj.dense_shape)
    triu = tf.sparse.transpose(tril)
    adj = tf.sparse.add(tril, triu)
    adj = tf.sparse.reorder(adj)  # pylint: disable=no-value-for-parameter
    return adj


@register
def normalized_adjacency(x: tf.SparseTensor, symmetric: bool = True) -> tf.SparseTensor:
    d = tf.sparse.reduce_sum(x, axis=0)
    if symmetric:
        d = tf.math.rsqrt(d)
        row, col = tf.unstack(x.indices, axis=1)
        x = x.with_values(
            x.values * tf.gather(d, row, axis=0) * tf.gather(d, col, axis=0)
        )
    else:
        x = x.with_values(x.values / tf.gather(d, x.indices[:, 0], axis=0))
    return x


@configurable
def normalized_laplacian(
    x: tf.SparseTensor,
    symmetric: bool = True,
    shift: float = 0.0,
    diag_always_one: bool = True,
) -> tf.SparseTensor:
    d = tf.sparse.reduce_sum(x, axis=0)
    if symmetric:
        d_rsqrt = tf.math.rsqrt(d)
        row, col = tf.unstack(x.indices, axis=1)
        x = x.with_values(
            -x.values
            * tf.gather(d_rsqrt, row, axis=0)
            * tf.gather(d_rsqrt, col, axis=0)
        )
    else:
        x = x.with_values(-x.values / tf.gather(d, x.indices[:, 0], axis=0))
    diag = tf.sparse.eye(x.dense_shape[0], dtype=x.dtype)
    if not diag_always_one:
        diag = diag.with_values(
            tf.where(
                d == 0, tf.zeros_like(d, dtype=x.dtype), tf.ones_like(d, dtype=x.dtype)
            )
        )
    return tf.sparse.add(diag.with_values(diag.values * (1 + shift)), x)


@configurable
def to_symmetric(x: tf.SparseTensor, half: bool = False) -> tf.SparseTensor:
    xt = tf.sparse.reorder(  # pylint: disable=no-value-for-parameter
        tf.sparse.transpose(x)
    )
    x = tf.sparse.add(x, xt)
    if half:
        x = x.with_values(x.values / 2)
    return x


@configurable
def remove_diag(x: tf.SparseTensor) -> tf.SparseTensor:
    row, col = tf.unstack(x.indices, axis=1)
    mask = row != col
    return tf.SparseTensor(
        tf.boolean_mask(x.indices, mask), tf.boolean_mask(x.values, mask), x.dense_shape
    )


@register
def actually_none(_: tp.Any):
    return None


@configurable
def to_format(x: tp.Union[tf.Tensor, tf.SparseTensor], fmt: str):
    if isinstance(x, tf.SparseTensor):
        if fmt == "dense":
            return tf.sparse.to_dense(x)
        assert fmt == "sparse"
        return x
    assert tf.is_tensor(x)
    if fmt == "dense":
        return x

    assert fmt == "sparse"
    return tf.sparse.from_dense(x)


@configurable
def row_normalize(x: tp.Union[tf.Tensor, tf.SparseTensor]):
    if isinstance(x, tf.Tensor):
        factor = tf.reduce_sum(x, axis=1, keepdims=True)
        factor = tf.where(factor == 0, tf.ones_like(factor), factor)
        return x / factor
    assert isinstance(x, tf.SparseTensor)
    row = x.indices[:, 0]
    factor = tf.math.segment_sum(x.values, row)
    return x.with_values(x.values / tf.gather(factor, row, axis=0))


def johnson_lindenstrauss_min_dim(n_samples: int, *, eps: float) -> int:
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return int((4 * np.log(n_samples) / denominator))


@register
def random_projection(
    features: tp.Union[tf.Tensor, tf.SparseTensor],
    k_or_eps: tp.Union[int, float],
    seed: tp.Optional[int] = None,
) -> tf.Tensor:
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)
    # num_nodes, n = features.shape
    n = features.shape[1]
    if isinstance(k_or_eps, int):
        k = k_or_eps
    else:
        assert isinstance(k_or_eps, float)
        k = johnson_lindenstrauss_min_dim(features.shape[1], eps=k_or_eps)
    if isinstance(features, tf.SparseTensor):
        R = rng.normal((n, k), stddev=1 / np.sqrt(k)).numpy()
        return tf.sparse.sparse_dense_matmul(features, R)
        # def project_col(col):
        #     return tf.squeeze(
        #         tf.sparse.sparse_dense_matmul(features, tf.expand_dims(col, axis=1)),
        #         axis=1,
        #     )

        # @tf.function
        # def project(R):
        #     return tf.map_fn(
        #         project_col,
        #         R,
        #         fn_output_signature=tf.TensorSpec(
        #             shape=(num_nodes,), dtype=features.dtype
        #         ),
        #     )

        # return tf.transpose(project(R))
    R = rng.normal((n, k), stddev=1 / np.sqrt(k))
    return tf.linalg.matmul(features, R)


# def _page_rank_propagate(
#     A: tf.SparseTensor,
#     epsilon: float,
#     x: tf.Tensor,
#     x0: tf.Tensor,
#     tol: float,
#     max_iter: int,
#     show_progress: bool,
# ) -> tf.Tensor:
#     if epsilon == 1:
#         return x
#     assert 0 < epsilon < 1, epsilon
#     I = tf.sparse.eye(A.shape[0], dtype=A.dtype)
#     L = tf.sparse.add(I, A.with_values(A.values * (epsilon - 1)))

#     if show_progress:
#         L = scipy_utils.to_scipy(L)

#         sol = np.zeros(x.shape, dtype=x.dtype.as_numpy_dtype)
#         for i in tqdm.trange(x.shape[1], desc="Computing page-rank vectors"):
#             s, _ = la.cg(L, x[:, i], x0=x0, tol=tol, maxiter=max_iter)
#             sol[:, i] = s

#         return tf.convert_to_tensor(epsilon * sol, x.dtype)

#     L = LinearOperatorSparseMatrix(L, is_self_adjoint=True, is_positive_definite=True)

#     sol = LinearOperatorCGSolver(L, x0=x0, tol=tol, max_iter=max_iter) @ x
#     return epsilon * sol


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


def _cg_solver_matrix(
    L: tp.Union[tf.SparseTensor, Matrix],
    *,
    show_progress: bool = False,
    parallel_iterations: tp.Optional[int] = None,
    **kwargs,
):
    if isinstance(L, tf.SparseTensor):
        L = SparseMatrix(L, is_self_adjoint=True, is_positive_definite=True)
    elif isinstance(L, tf.Tensor):
        L = FullMatrix(L, is_self_adjoint=True, is_positive_definite=True)
    else:
        assert isinstance(L, Matrix), type(L)
        assert L.is_self_adjoint
        assert L.is_positive_definite
    solver = CGSolverMatrix(L, **kwargs)

    if parallel_iterations and show_progress:
        raise Exception("Cannot use both `parallel_iterations` and `show_progress`")

    if parallel_iterations:
        solver = MappedMatrix(solver, parallel_iterations)
    if show_progress:
        solver = ProgMatrix(solver)
    return solver


def heat_matrix(
    A: tp.Union[tf.SparseTensor, Matrix],
    *,
    t: float = 1.0,
    symmetric: bool = True,
    renormalized: bool = False,
    show_progress: bool = False,
    preprocess: bool = False,
) -> tp.Union[Matrix, tf.Tensor]:
    if renormalized:
        I = tf.sparse.eye(A.shape[0], dtype=A.dtype)
        A = tf.sparse.add(A, I)
    L = normalized_laplacian(A, symmetric=symmetric)
    L = L.with_values(-t * L.values)
    mat = ExponentialMatrix(SparseMatrix(L, is_self_adjoint=True))
    if show_progress:
        mat = ProgMatrix(mat)
    if preprocess:
        mat = mat.to_dense()
    return mat


@register
def heat_propagate(
    adj: tf.SparseTensor,
    x: tf.Tensor,
    t: tp.Union[float, tp.Iterable[float]],
    symmetric: bool = True,
    renormalized: bool = False,
    show_progress: bool = True,
) -> tf.Tensor:
    if isinstance(t, (int, float)):
        t = (t,)

    features = [
        heat_matrix(
            adj,
            t=ti,
            show_progress=show_progress,
            symmetric=symmetric,
            renormalized=renormalized,
        )
        @ x
        for ti in t
    ]
    if len(features) == 1:
        return features[0]
    return tf.concat(features, 1)


def page_rank_matrix(
    A: tp.Union[tf.SparseTensor, Matrix],
    *,
    epsilon: float = 0.1,
    tol: float = 1e-5,
    max_iter: int = 20,
    symmetric: bool = True,
    renormalized: bool = False,
    show_progress: bool = False,
    preprocess: bool = False,
    parallel_iterations: tp.Optional[int] = None,
    unscaled: bool = False,
    rescale_factor: float = 1.0,
) -> tp.Union[Matrix, tf.Tensor]:
    if not unscaled:
        rescale_factor *= epsilon
    I = tf.sparse.eye(A.shape[0], dtype=A.dtype)
    if renormalized:
        A = tf.sparse.add(A, I)
    if symmetric:
        A = normalize_sparse(A, symmetric=True)
        L = tf.sparse.add(I, A.with_values(A.values * (epsilon - 1)))
        mat = _cg_solver_matrix(
            L,
            show_progress=show_progress,
            tol=tol,
            max_iter=max_iter,
            parallel_iterations=parallel_iterations,
        )
        if rescale_factor != 1.0:
            mat = ScaledIdentityMatrix(mat.shape[0], rescale_factor) @ mat
    else:
        d = tf.sparse.reduce_sum(A, axis=1)
        disconnected = d == 0
        ones = tf.ones_like(d)
        # add self-loop for completely disconnected nodes
        d = tf.where(disconnected, ones, d)
        L = tf.sparse.add(I.with_values(d), A.with_values(A.values * (1 - epsilon)))
        mat = _cg_solver_matrix(
            L,
            show_progress=show_progress,
            parallel_iterations=parallel_iterations,
            tol=tol,
            max_iter=max_iter,
        )
        if rescale_factor != 1.0:
            raise NotImplementedError
            mat = CompositionMatrix(
                (DiagMatrix(tf.where(disconnected, ones, ones * epsilon)), mat)
            )

    def _to_dense(mat: Matrix):

        if isinstance(mat, CompositionMatrix):
            # LinearOperatorComposition._to_dense implementation is terrible
            return mat.operators[0] @ _to_dense(composition_matrix(*mat.operators[1:]))
        return mat.to_dense()

    if preprocess:
        return _to_dense(mat)
    return mat


@register
def page_rank_propagate(
    adj: tf.SparseTensor,
    x: tf.Tensor,
    epsilon: tp.Union[float, tp.Iterable[float]],
    tol: float = 1e-5,
    max_iter: int = 1000,
    show_progress: bool = True,
    parallel_iterations: tp.Optional[int] = None,
    renormalized: bool = False,
    unscaled: bool = False,
    rescale_factor: float = 1.0,
) -> tf.Tensor:
    if isinstance(epsilon, (int, float)):
        epsilon = (epsilon,)

    features = [
        page_rank_matrix(
            adj,
            epsilon=eps,
            tol=tol,
            max_iter=max_iter,
            show_progress=show_progress,
            renormalized=renormalized,
            unscaled=unscaled,
            parallel_iterations=parallel_iterations,
            rescale_factor=rescale_factor,
        )
        @ x
        for eps in epsilon
    ]
    if len(features) == 1:
        return features[0]
    return tf.concat(features, 1)


# @register
# def page_rank_propagate(
#     adj: tf.SparseTensor,
#     x: tf.Tensor,
#     epsilon: tp.Union[float, tp.Iterable[float]],
#     tol: float = 1e-5,
#     max_iter: int = 1000,
#     show_progress: bool = True,
# ) -> tf.Tensor:
#     adj.shape.assert_has_rank(2)
#     dtype = adj.dtype
#     assert dtype.is_floating
#     x = tf.convert_to_tensor(x, dtype=dtype)
#     x.shape.assert_has_rank(2)
#     row_sum = tf.sparse.reduce_sum(adj, axis=1)
#     x0 = tf.sqrt(row_sum) / tf.reduce_sum(row_sum).numpy()
#     adj = normalize_sparse(adj, symmetric=True)
#     if isinstance(epsilon, float):
#         epsilon = (epsilon,)
#     else:
#         assert len(epsilon) > 0

#     out = [
#         _page_rank_propagate(adj, eps, x, x0, tol, max_iter, show_progress)
#         for eps in epsilon
#     ]

#     if len(out) == 1:
#         return out[0]
#     return tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
#         out, axis=-1
#     )


def _page_rank_labels_propagate(
    A: tf.SparseTensor,
    epsilon: float,
    x0: tf.Tensor,
    labels: tf.Tensor,
    num_classes: int,
    train_ids: tf.Tensor,
    tol: float,
    max_iter: int,
    show_progress: bool,
) -> tf.Tensor:
    assert 0 < epsilon < 1, epsilon
    num_nodes = A.shape[0]
    num_labels = train_ids.shape[0]
    assert labels.shape == (num_labels,), (labels.shape, num_labels)
    dtype = A.dtype
    assert dtype.is_floating
    I = tf.sparse.eye(num_nodes, dtype=dtype)
    L = tf.sparse.add(I, A.with_values(A.values * (epsilon - 1)))
    if show_progress:
        L = scipy_utils.to_scipy(L)
        labels = labels.numpy()
        train_ids = train_ids.numpy()
        out = np.zeros((num_nodes, num_classes), dtype=dtype.as_numpy_dtype)
        for i in tqdm.trange(num_labels, desc="Computing label page-rank vectors"):
            x = np.zeros((num_nodes,), dtype=dtype.as_numpy_dtype)
            train_id = train_ids[i]
            label = labels[i]
            s, _ = la.cg(L, x, x0=x0, tol=tol, maxiter=max_iter)
            s[train_id] = 0
            out[:, label] += s

        return tf.convert_to_tensor(epsilon * out, L.dtype)

    L = LinearOperatorSparseMatrix(L, is_self_adjoint=True, is_positive_definite=True)
    indices = tf.stack(
        (train_ids, tf.range(num_labels, dtype=train_ids.dtype)), axis=-1
    )
    x = tf.scatter_nd(
        indices,
        tf.ones((num_labels,), dtype=dtype),
        (num_nodes, num_labels),
    )  # [N, L]
    sol = LinearOperatorCGSolver(L, x0=x0, max_iter=max_iter, tol=tol) @ x  # [N, L]
    diag = tf.gather_nd(sol, indices)  # [L]
    solT = tf.transpose(sol)  # [L, N]
    solT_reduced = tf.math.unsorted_segment_sum(
        solT, tf.gather(labels, train_ids), num_classes
    )  # [C, N]
    sol_reduced = tf.transpose(solT_reduced)  # [N, C]
    sol = sol_reduced - tf.scatter_nd(
        tf.stack((train_ids, tf.cast(labels, train_ids.dtype)), axis=1),
        diag,
        (num_nodes, num_classes),
    )
    return epsilon * sol


@register
def page_rank_labels_propagate(
    adj: tf.SparseTensor,
    labels: tf.Tensor,
    train_ids: tf.Tensor,
    epsilon: tp.Union[float, tp.Iterable[float]],
    tol: float = 1e-5,
    max_iter: int = 1000,
    show_progress: bool = True,
) -> tf.Tensor:
    adj.shape.assert_has_rank(2)
    dtype = adj.dtype
    assert dtype.is_floating
    assert labels.shape == train_ids.shape, (labels.shape, train_ids.shape)
    row_sum = tf.sparse.reduce_sum(adj, axis=1)
    x0 = tf.sqrt(row_sum) / tf.reduce_sum(row_sum).numpy()
    adj = normalize_sparse(adj, symmetric=True)
    if isinstance(epsilon, float):
        epsilon = (epsilon,)
    else:
        assert len(epsilon) > 0

    num_classes = tf.reduce_max(labels).numpy() + 1
    out = [
        _page_rank_labels_propagate(
            adj, eps, x0, labels, num_classes, train_ids, tol, max_iter, show_progress
        )
        for eps in epsilon
    ]
    if len(out) == 1:
        return out[0]
    return tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        out, axis=-1
    )

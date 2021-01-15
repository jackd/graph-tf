from typing import Tuple, Union

import tensorflow as tf

from graph_tf.utils.type_checks import is_sparse_tensor


def pad_to_size(x, size, axis=0, leading=0):
    size = tf.convert_to_tensor(size)
    x = tf.convert_to_tensor(x)
    paddings = [[0, 0] for _ in range(x.shape.ndims)]
    paddings[axis] = [leading, size - leading - tf.shape(x, out_type=size.dtype)[axis]]
    out = tf.pad(x, paddings)  # pylint: disable=no-value-for-parameter
    # set static shape
    size = tf.get_static_value(size)
    if size is not None:
        out_shape = list(x.shape)
        out_shape[axis] = size
        out.set_shape(out_shape)
    return out


def arnoldi_iteration(
    A: Union[tf.Tensor, tf.SparseTensor],
    b: tf.Tensor,
    n: int,
    symmetric: bool = False,
    eps: float = 1e-12,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes a basis of the (n + 1)-Krylov subspace of A for each col of b.

    The Krylov-subspace is the space spanned by {b, Ab, ..., A^n b}.

    Based on np implementation at

    Arguments
        A: [m, m] float Tensor or SparseTensor
        b: [m, p] initial Tensor
        n: dimension of Krylov subspace, must be >= 1
        symmetric: if true, A will be assumed to be symmetric, hence only the previous
            two components will be removed in re-orthogonalization.
        eps: threshold to exit early

    Returns
      Q: [m , (n + 1), p] float Tensor, the columns are an orthonormal basis of the
        Krylov subspace.
      h: [(n + 1), n, p] float Tensor, A on basis Q. It is upper Hessenberg.
    """
    assert b.dtype == A.dtype, "dtypes must be the same"
    if is_sparse_tensor(A):
        matmul = tf.sparse.sparse_dense_matmul
    else:
        matmul = tf.linalg.matmul

    if b.dtype.is_complex:

        def conjugate(x):
            return tf.complex(tf.math.real(x), -tf.math.imag(x))

    else:

        def conjugate(x):
            return x

    def dot(x, y):
        return tf.einsum("dp,dp->p", x, y)
        # return tf.squeeze(tf.tensordot(x, y, (0, 0)), axis=0)

    h = []
    q = b / tf.linalg.norm(b, axis=0)  # Normalize the input vector
    Q = [q]
    z = tf.zeros_like(b)

    for _ in range(n):
        v = matmul(A, q)  # Generate a new candidate vector
        hk = []
        h.append(hk)
        for qi in Q[-2:] if symmetric else Q:
            hjk = dot(conjugate(qi), v)
            hk.append(hjk)
            v = v - hjk * qi

        h_final = tf.linalg.norm(v, axis=0)
        hk.append(h_final)
        q = tf.where(h_final > eps, v / h_final, z)
        Q.append(q)

    Q = tf.stack(Q, axis=1)
    if symmetric:
        leading = [max(0, i - 1) for i in range(n)]
    else:
        leading = [0] * n
    h = [pad_to_size(hi, n + 1, leading=l, axis=0) for l, hi in zip(leading, h)]

    h = tf.stack(h, axis=1)
    return Q, h


def lanczos_iteration(
    A: Union[tf.Tensor, tf.SparseTensor], b: tf.Tensor, n: int, eps: float = 1e-12
):
    """
    Computes a basis of the (n + 1)-Krylov subspace of symmetric A for each col of b.

    The Krylov-subspace is the space spanned by {b, Ab, ..., A^n b}.

    See https://chen.pw/research/cg/arnoldi_lanczos.html

    Arguments
        A: [m, m] float Tensor or SparseTensor, assumed symmetric.
        b: [m, p] initial Tensor.
        n: dimension of Krylov subspace, must be >= 1
        eps: threshold to exit early.

    Returns
        Q: [m , (n + 1), p] float Tensor, the columns are an orthonormal basis of the
          Krylov subspace.
        d: [n, p] diagonal values of H.
        l: [n, p] sub-diagonal values of H.
    """
    assert b.dtype == A.dtype, "dtypes must be the same"
    if is_sparse_tensor(A):
        matmul = tf.sparse.sparse_dense_matmul
    else:
        matmul = tf.linalg.matmul

    if b.dtype.is_complex:

        def conjugate(x):
            return tf.complex(tf.math.real(x), -tf.math.imag(x))

    else:

        def conjugate(x):
            return x

    def dot(x, y):
        return tf.einsum("dp,dp->p", x, y)
        # return tf.squeeze(tf.tensordot(x, y, (0, 0)), axis=0)

    q = b / tf.linalg.norm(b, axis=0)  # Normalize the input vector
    Q = [q]
    z = tf.zeros_like(b)
    d = []
    l = []

    for _ in range(n):
        v = matmul(A, q)  # Generate a new candidate vector
        if len(Q) >= 2:
            v = v - l[-1] * Q[-2]

        di = dot(conjugate(Q[-1]), v)
        d.append(di)
        v = v - di * Q[-1]

        li = tf.linalg.norm(v, axis=0)
        l.append(li)
        q = tf.where(li > eps, v / li, z)

        Q.append(q)

    Q = tf.stack(Q, axis=1)
    d = tf.stack(d)
    l = tf.stack(l)
    return Q, d, l


def ritz_embedding(laplacian: tf.SparseTensor, x0: tf.Tensor, output_size: int):
    Q, d, l = lanczos_iteration(laplacian, x0, output_size)
    diags = tf.stack(
        (
            tf.pad(l, [[0, 1]]),  # pylint:disable=no-value-for-parameter
            d,
            tf.pad(l, [[1, 0]]),  # pylint:disable=no-value-for-parameter
        ),
        axis=0,
    )
    tri_diag = tf.linalg.LinearOperatorTridiag(diags, is_self_adjoint=True)
    vals, vecs = tf.linalg.eigh(tri_diag.to_dense())
    ritz_features = tf.matmul(Q, vecs)
    return ritz_features, vals

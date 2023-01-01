import functools

import gin
import tensorflow as tf

from graph_tf.utils.ops import normalize_sparse

register = functools.partial(gin.register, module="gtf.ssgc.transforms")


@register
def ssgc_transform(
    A: tf.SparseTensor,
    features: tf.Tensor,
    alpha: float,
    degree: int,
    *,
    symmetric: bool = True,
    renormalized: bool = True,
) -> tf.Tensor:
    if renormalized:
        A = tf.sparse.add(A, tf.sparse.eye(A.shape[0], dtype=A.dtype))
    A = normalize_sparse(A, symmetric=symmetric)
    x = features
    terms = []
    for _ in range(degree):
        x = tf.sparse.sparse_dense_matmul(A, x)
        terms.append(x)
    return alpha * features + (1 - alpha) * tf.add_n(terms) / degree

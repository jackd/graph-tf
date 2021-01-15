from typing import Optional

import tensorflow as tf


def multi_attention_v0(
    features: tf.Tensor, attention: tf.Tensor, adjacency: tf.SparseTensor
):
    """
    Implementation using unstack / stack / sparse_dense_matmul

    Args:
        features: [Ni, H, F]
        attention: [E, H]
        adjacency: [No, Ni], E non-zero entries.

    Returns:
        [No, H, F] features.
    """
    features = [
        tf.sparse.sparse_dense_matmul(adjacency.with_values(attn), f)
        for attn, f in zip(tf.unstack(attention, axis=1), tf.unstack(features, axis=1))
    ]
    return tf.stack(features, axis=1)


def multi_attention_v1(
    features: tf.Tensor,
    attention: tf.Tensor,
    adjacency: tf.SparseTensor,
    reduction: Optional[str] = None,
):
    """
    Implementation using gather / segment sum.

    Args:
        features: [Ni, H, F].
        attention: [E, H].
        adjacency: [No, Ni], E non-zero entries.
        reduction: None, "sum" or "mean".

    Returns:
        [No, H, F] features, or [No, F] if reduction is not None.
    """
    row, col = tf.unstack(adjacency.indices, axis=-1)
    features = tf.gather(features, col, axis=0)
    if reduction is None:
        features = features * tf.expand_dims(attention, axis=-1)
    else:
        # features = tf.einsum('ehf,eh->ef')
        features = tf.linalg.matvec(features, attention, transpose_a=True)
        if reduction == "mean":
            features = features / attention.shape[1]
        else:
            assert reduction == "sum"
    return tf.math.segment_sum(features, row)

import tensorflow as tf


def spectral_graph_conv_v1(X: tf.Tensor, w: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
    """
    Perform a spectral graph convolution.

    For a single input/output channel, this corresponds to `V @ diag(w) @ V.T @ X`. We
    map this over all input and output channels and sum over the resulting input
    channel dimension.

    Args:
        X: [N, fi] input features
        w: [e, fi, fo] kernel
        V: [N, e] eigenvectors

    Returns:
        [N, fo] output features
    """
    assert w.shape[:2] == (V.shape[1], X.shape[1]), (X.shape, w.shape, V.shape)
    return tf.einsum("ne,eio,ei->no", V, w, tf.matmul(V, X, transpose_a=True))


def spectral_graph_conv_v2(X: tf.Tensor, w: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
    assert w.shape[:2] == (V.shape[1], X.shape[1]), (X.shape, w.shape, V.shape)
    return tf.einsum("ne,eio,me,mi->no", V, w, V, X)


spectral_graph_conv_v2.__doc__ = spectral_graph_conv_v1.__doc__

spectral_graph_conv = spectral_graph_conv_v1


def channelwise_spectral_graph_conv(
    X: tf.Tensor, w: tf.Tensor, V: tf.Tensor
) -> tf.Tensor:
    """
    Perform a channelwise spectral grpah convolution, `V @ diag(w) @ V.T @ X`.
    """
    X = tf.convert_to_tensor(X)
    w = tf.convert_to_tensor(w)
    V = tf.convert_to_tensor(V)
    X.shape.assert_has_rank(2)
    w.shape.assert_has_rank(1)
    V.shape.assert_has_rank(2)
    assert w.shape[0] == V.shape[1], (w.shape, V.shape)
    return tf.matmul(V, tf.expand_dims(w, axis=1) * tf.matmul(V, X, transpose_a=True))


def depthwise_spectral_graph_conv(
    X: tf.Tensor, w: tf.Tensor, V: tf.Tensor
) -> tf.Tensor:
    """
    Performs a depthwise spectral graph conv, analagous to tf's DepthwiseConv.

    Args:
        X: [N, f]
        w: [e, f*depth_multiplier]
        V: [N, e]

    Returns:
        [N, f * depth_multiplier]
    """
    X = tf.convert_to_tensor(X)
    w = tf.convert_to_tensor(w)
    V = tf.convert_to_tensor(V)
    X.shape.assert_has_rank(2)
    V.shape.assert_has_rank(2)
    w.shape.assert_has_rank(2)

    e, fd = w.shape
    f = X.shape[1]
    assert w.shape[0] == V.shape[1], (w.shape, V.shape)
    assert fd % f == 0, (fd, f)
    d = fd // f
    w = tf.reshape(w, (e, f, d))

    wvx = tf.reshape(
        w * tf.expand_dims(tf.matmul(V, X, transpose_a=True), 2), (e, f * d)
    )
    return tf.matmul(V, wvx)


def separable_spectral_graph_conv(
    X: tf.Tensor, w: tf.Tensor, V: tf.Tensor, pointwise_kernel: tf.Tensor
) -> tf.Tensor:
    """
    Performs a separable spectral graph convolution, analagous to tf's SeparableConv.

    Args:
        X: [N, fi]
        w: [e, fi * depth_multiplier]
        V: [N, e]
        pointwise_kernel: [fi * depth_multiplier, fo]
    """
    X = tf.convert_to_tensor(X)
    w = tf.convert_to_tensor(w)
    V = tf.convert_to_tensor(V)
    pointwise_kernel = tf.convert_to_tensor(pointwise_kernel)

    X.shape.assert_has_rank(2)
    w.shape.assert_has_rank(2)
    V.shape.assert_has_rank(2)
    pointwise_kernel.shape.assert_has_rank(2)

    fi = X.shape[1]
    e, fd = w.shape
    assert fd % fi == 0, (fd, fi)
    d = fd // fi
    assert pointwise_kernel.shape[0] == w.shape[1], (pointwise_kernel.shape, w.shape)

    w = tf.reshape(w, (e, fi, d))
    wvx = tf.reshape(
        w * tf.expand_dims(tf.matmul(V, X, transpose_a=True), 2), (e, fi * d)
    )
    # presumably e << n, so perform multiplication right-to-left.
    return tf.matmul(V, tf.matmul(wvx, pointwise_kernel))

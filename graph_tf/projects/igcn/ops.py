import typing as tp
import warnings

import tensorflow as tf

from graph_tf.utils.linalg import SparseLinearOperator


class CGResult(tp.NamedTuple):
    i: tf.Tensor
    x: tf.Tensor
    r: tf.Tensor
    p: tf.Tensor
    gamma: tf.Tensor


def sparse_cg(
    A: tp.Union[SparseLinearOperator, tf.SparseTensor],
    rhs: tf.Tensor,
    *,
    tol: float = 1e-5,
    max_iter: int = 20,
) -> CGResult:
    r"""
    Conjugate gradient method for `tf.SparseTensor` coefficients.

    Args:
        A:
        rhs: rank 1 or 2

    Returns:
        output: A namedtuple representing the final state with fields:
        - i: A scalar `int32` `Tensor`. Number of iterations executed.
        - x: A rank-1 `Tensor` of shape `[..., N]` containing the computed
            solution.
        - r: A rank-1 `Tensor` of shape `[.., M]` containing the residual vector.
        - p: A rank-1 `Tensor` of shape `[..., N]`. `A`-conjugate basis vector.
        - gamma: \\(r \dot M \dot r\\), equivalent to  \\(||r||_2^2\\) when
            `preconditioner=None`.

        Note: if `rhs` is rank 2, the results are stacked along an additional final
        axis.
    """
    if not isinstance(A, SparseLinearOperator):
        A = SparseLinearOperator(A, is_self_adjoint=True, is_positive_definite=True)
    rhs = tf.convert_to_tensor(rhs)

    def fn(rhs):
        return tf.linalg.experimental.conjugate_gradient(
            A, rhs, tol=tol, max_iter=max_iter
        )

    if rhs.shape.ndims == 2:
        sols = [fn(r) for r in tf.unstack(rhs, axis=1)]
        return CGResult(
            tf.stack([sol.i for sol in sols], axis=-1),
            tf.stack([sol.x for sol in sols], axis=-1),
            tf.stack([sol.r for sol in sols], axis=-1),
            tf.stack([sol.p for sol in sols], axis=-1),
            tf.stack([sol.gamma for sol in sols], axis=-1),
        )
    rhs.shape.assert_has_rank(1)
    return CGResult(*fn(rhs))


def _preprocess_preds(output, from_logits: bool):
    if hasattr(output, "_keras_logits"):
        output = output._keras_logits  # pylint: disable=protected-access
        if from_logits:
            warnings.warn(
                '"`sparse_categorical_crossentropy` received `from_logits=True`, but '
                "the `output` argument was produced by a sigmoid or softmax "
                'activation and thus does not represent logits. Was this intended?"',
                stacklevel=2,
            )
            from_logits = True
    elif (
        not from_logits
        and not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable))
        and output.op.type == "Softmax"
    ) and not hasattr(output, "_keras_history"):
        # When softmax activation function is used for output operation, we
        # use logits from the softmax function directly to compute loss in order
        # to prevent collapsing zero when training.
        # See b/117284466
        assert len(output.op.inputs) == 1
        output = output.op.inputs[0]
        from_logits = True
    elif not from_logits:
        epsilon_ = tf.convert_to_tensor(
            tf.keras.backend.epsilon(), output.dtype.base_dtype
        )
        output = tf.clip_by_value(output, epsilon_, 1 - epsilon_)
        output = tf.math.log(output)
    return output


def quadratic_sparse_categorical_crossentropy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    from_logits: bool = True,
) -> tf.Tensor:
    y_pred = _preprocess_preds(y_pred, from_logits=from_logits)
    assert y_true.dtype.is_integer, (y_true.shape, y_true.dtype)
    y_true.shape.assert_has_rank(1)
    assert y_pred.dtype.is_floating
    y_pred.shape.assert_has_rank(2)

    y_pred = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)

    f1 = -tf.gather(y_pred, y_true, batch_dims=1)
    f2 = tf.reduce_mean(y_pred**2, axis=1)
    return f1 + f2 / 2

    # xs = tf.reduce_sum(y_pred, axis=1)
    # correct = tf.gather(y_pred, y_true, batch_dims=1)
    # f1 = -correct + xs / num_classes
    # f2 = tf.reduce_sum(y_pred ** 2, axis=1) / num_classes - xs ** 2 / num_classes ** 2
    # return f1 + f2 / 2


def lazy_quadratic_categorical_crossentropy(
    linear_factor: tf.Tensor,
    quadratic_factor: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """
    Quadratic approximations of sparse x-entropy loss where `logits = A @ y_pred`.

    Note that this implementation computes `at.T @ y_pred`, where `at` is either
    `A.T` or it's random projection `(R @ A).T`. Using the former largely defeats the
    purpose of using this implementation (the point is to avoid `N` computation times.

    Note this implementation explicitly computes `at.T @ y_pred` so isn't really lazy.
    but since `at` can either be `A.T`

    A represents a [N, B] tensor - presumably B << N, otherwise eager evaluation is more
    efficient.

    Result is the sum over the [N] dimension.

    Args:
        linear_factor: [B, C] `A.T @ Y`, where `Y` is the one-hot encoding of labels.
        quadratic_factor: [B, N or K] `A.T` or it's `K`-dimensional random projection.
        y_pred: [B, C]

    Returns:
        scalar loss without the constant L * np.log(C) term.
    """
    assert y_pred.dtype.is_floating
    y_pred.shape.assert_has_rank(2)
    assert linear_factor.shape == y_pred.shape, (linear_factor.shape, y_pred.shape)
    num_classes = tf.convert_to_tensor(y_pred.shape[1], dtype=y_pred.dtype)

    y_pred = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
    f1 = -tf.reduce_sum(linear_factor * y_pred)
    f2 = (
        tf.reduce_sum(tf.linalg.matmul(quadratic_factor, y_pred, adjoint_a=True) ** 2)
        / num_classes
    )
    return f1 + f2 / 2


def lazy_quadratic_categorical_crossentropy_v2(
    linear_factor: tf.Tensor,
    quadratic_factor: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """
        Quadratic approximations of sparse x-entropy loss where `logits = A @ y_pred`.

        Note `A @ y_pred` is never formed, and `A` need never be formed explicitly.

        A represents a [N, B] tensor - presumably B << N, otherwise eager evaluation is
        more efficient.

        Result is the sum over the [N] dimension.

        This is an alternative implementation to
        `lazy_quadratic_categorical_crossentropy` which requires precomputing
        `A.T @ A` and is O(B^2 * C) rather than O(B K C).

        Args:
            num_labels: [] N
            linear_factor: [B, C] `A.T @ Y`, where `Y` is the one-hot encoding of
                labels.
            quadratic_factor: [B, B] `A.T @ A`
            y_pred: [B, C]

    Returns:
        scalar loss without the constant L * np.log(C) term.
    """
    assert y_pred.dtype.is_floating
    y_pred.shape.assert_has_rank(2)
    assert linear_factor.shape == y_pred.shape, (linear_factor.shape, y_pred.shape)
    num_classes = tf.convert_to_tensor(y_pred.shape[1], dtype=y_pred.dtype)

    y_pred = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
    f1 = -tf.reduce_sum(linear_factor * y_pred)
    f2 = (
        tf.reduce_sum(
            quadratic_factor * tf.linalg.matmul(y_pred, y_pred, adjoint_b=True)
        )
        / num_classes
    )
    return f1 + f2 / 2


def linear_categorical_crossentropy(linear_factor: tf.Tensor, y_pred: tf.Tensor):
    """Linearized categorical crossentropy without the constant factor."""
    assert y_pred.dtype.is_floating
    y_pred.shape.assert_has_rank(2)
    assert linear_factor.shape == y_pred.shape, (linear_factor.shape, y_pred.shape)
    y_pred = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
    return -tf.reduce_sum(linear_factor * y_pred, axis=1)


def hacky_quadratic_categorical_crossentropy(
    linear_factor: tf.Tensor, y_pred: tf.Tensor, num_labels: tp.Optional[int] = None
) -> tf.Tensor:
    assert y_pred.dtype.is_floating
    y_pred.shape.assert_has_rank(2)
    assert linear_factor.shape == y_pred.shape, (linear_factor.shape, y_pred.shape)
    num_classes = tf.convert_to_tensor(y_pred.shape[1], dtype=y_pred.dtype)

    y_pred = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
    f1 = -tf.reduce_sum(linear_factor * y_pred, axis=1)
    f2 = tf.reduce_sum(y_pred**2, axis=1) / num_classes
    if num_labels:
        f1 = f1 / num_labels
        f2 = f2 / tf.cast(tf.shape(f2)[0], f2.dtype)
    return f1 + f2 / 2

import functools
import typing as tp

import gin
import tensorflow as tf
from tflo.matrix.core import Matrix

from graph_tf.projects.igcn import ops

register = functools.partial(gin.register, module="gtf.igcn.losses")


@register
class QuadraticSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = "auto",
        name: str = "quadratic_sparse_categorical_crossentropy",
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits

    def get_config(self):
        config = super().get_config()
        config["from_logits"] = self.from_logits
        return config

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if y_true.shape.ndims == 2:
            y_true = tf.squeeze(y_true, axis=1)
        if y_true.dtype.is_floating:
            y_true = tf.cast(y_true, tf.int64)
        return ops.quadratic_sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )


@register
class LinearCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(
        self,
        num_labels: tp.Optional[int] = None,
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "linear_categorical_crossentropy",
    ):
        self.num_labels = num_labels
        super().__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super().get_config()
        config["num_labels"] = self.num_labels
        return config

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = ops.linear_categorical_crossentropy(y_true, y_pred)
        if self.num_labels:
            loss = loss / self.num_labels
        return loss


@register
class HackyQuadraticCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(
        self,
        num_labels: tp.Optional[int] = None,
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "hacky_quadratic_categorical_crossentropy",
    ):
        self.num_labels = num_labels
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return ops.hacky_quadratic_categorical_crossentropy(
            y_true, y_pred, num_labels=self.num_labels
        )


class LazyQuadraticCrossentropyLabelData(  # pylint: disable=abstract-method
    tf.experimental.BatchableExtensionType
):
    linear_factor: tf.Tensor  # [B, C]
    quadratic_factor: tp.Union[tf.Tensor, Matrix]  # [B, K]

    @property
    def shape(self):
        return self.linear_factor.shape

    @property
    def dtype(self):
        return self.linear_factor.dtype

    def to_v2(self) -> "LazyQuadraticCrossentropyV2LabelData":
        return LazyQuadraticCrossentropyV2LabelData(
            self.linear_factor,
            tf.matmul(self.quadratic_factor, self.quadratic_factor, transpose_b=True),
        )

    class Spec:
        linear_factor: tf.TensorSpec
        quadratic_factor: tf.TypeSpec

        @property
        def dtype(self):
            return self.linear_factor.dtype

        @property
        def shape(self):
            return self.linear_factor.shape


@tf.experimental.dispatch_for_api(tf.cast, {"x": LazyQuadraticCrossentropyLabelData})
def _cast(x: LazyQuadraticCrossentropyLabelData, dtype: tf.DType, name=None):
    assert dtype.is_floating, dtype
    with tf.name_scope(name or "Cast"):
        return LazyQuadraticCrossentropyLabelData(
            tf.cast(x.linear_factor, dtype),
            tf.cast(x.quadratic_factor, dtype),
        )


@tf.experimental.dispatch_for_api(
    tf.shape, {"input": LazyQuadraticCrossentropyLabelData}
)
def _shape(
    input: LazyQuadraticCrossentropyLabelData,  # pylint: disable=redefined-builtin
    out_type: tf.DType = tf.int32,
    name=None,
) -> tf.Tensor:
    with tf.name_scope(name or "Shape"):
        return tf.shape(input.linear_factor, out_type=out_type)


def lazy_quadratic_crossentropy_label_data(
    A: tf.Tensor, labels: tf.Tensor, num_classes: int, R: tp.Optional[tf.Tensor] = None
) -> LazyQuadraticCrossentropyLabelData:
    A.shape.assert_has_rank(2)
    assert A.dtype.is_floating
    labels.shape.assert_has_rank(1)
    assert labels.dtype.is_integer
    linear_factor = tf.linalg.adjoint(A) @ tf.one_hot(labels, num_classes)
    if R is None:
        at = tf.linalg.adjoint(A)
    else:
        at = tf.matmul(R, A, adjoint_b=True)

    return LazyQuadraticCrossentropyV2LabelData(linear_factor, at)


@register
class LazyQuadraticCrossentropyV2LabelData(  # pylint: disable=abstract-method
    tf.experimental.ExtensionType
):
    linear_factor: tf.Tensor  # [B, C]
    quadratic_factor: tf.Tensor  # [B, B]

    @property
    def shape(self):
        return self.linear_factor.shape

    @property
    def dtype(self):
        return self.linear_factor.dtype


@tf.experimental.dispatch_for_api(tf.cast, {"x": LazyQuadraticCrossentropyV2LabelData})
def _cast(x: LazyQuadraticCrossentropyV2LabelData, dtype: tf.DType, name=None):
    assert dtype.is_floating, dtype
    with tf.name_scope(name or "Cast"):
        return LazyQuadraticCrossentropyV2LabelData(
            tf.cast(x.linear_factor, dtype),
            tf.cast(x.quadratic_factor, dtype),
        )


@tf.experimental.dispatch_for_api(
    tf.shape, {"input": LazyQuadraticCrossentropyV2LabelData}
)
def _shape(
    input: LazyQuadraticCrossentropyV2LabelData,  # pylint: disable=redefined-builtin
    out_type: tf.DType = tf.int32,
    name=None,
) -> tf.Tensor:
    with tf.name_scope(name or "Shape"):
        return tf.shape(input.linear_factor, out_type=out_type)


@register
class LazyQuadraticCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, num_labels: tp.Optional[int] = None, **kwargs):
        self.num_labels = num_labels
        if "reduction" in kwargs:
            assert kwargs["reduction"] == tf.keras.losses.Reduction.SUM
        else:
            kwargs["reduction"] = tf.keras.losses.Reduction.SUM
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["num_labels"] = self.num_labels
        return config

    def __call__(
        self,
        y_true: tp.Union[
            LazyQuadraticCrossentropyLabelData, LazyQuadraticCrossentropyV2LabelData
        ],
        y_pred: tf.Tensor,
        sample_weight: tp.Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        assert sample_weight is None, "not implemented"
        if isinstance(y_true, LazyQuadraticCrossentropyLabelData):
            loss = ops.lazy_quadratic_categorical_crossentropy(
                y_true.linear_factor, y_true.quadratic_factor, y_pred
            )
        elif isinstance(y_true, LazyQuadraticCrossentropyV2LabelData):
            loss = ops.lazy_quadratic_categorical_crossentropy_v2(
                y_true.linear_factor, y_true.quadratic_factor, y_pred
            )
        else:
            raise TypeError(
                "y_true must be LazyQuadraticCrossentropyLabelData or "
                f"LazyQuadraticCrossentropyV2LabelData, got {type(y_true)}"
            )
        if self.num_labels is not None:
            loss = loss / self.num_labels
        return loss

    def call(self, y_true, y_pred):
        raise RuntimeError("Should not be called. Use `__call__` instead.")

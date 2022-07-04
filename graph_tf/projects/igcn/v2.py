import typing as tp

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf
from tflo.matrix.core import CompositionMatrix, FullMatrix, Matrix

from graph_tf.data.single import DataSplit, SemiSupervisedSingle
from graph_tf.data.transforms import transformed
from graph_tf.projects.igcn.data import get_shifted_laplacian, sparse_cg_solver
from graph_tf.projects.igcn.scalable import losses


def get_eig_decomp(adjacency: tf.SparseTensor, k: int = 32):
    n = adjacency.shape[0]
    row, col = adjacency.indices.numpy().T
    data = adjacency.values.numpy()
    adj = sp.coo_matrix((data, (row, col)), shape=adjacency.shape)
    d = np.array(adj.sum(axis=1)).squeeze(axis=1)
    d = d ** (-0.5)
    data = data * d[row] * d[col]
    shifted_lap = (
        -sp.identity(n, dtype=np.float32)
        - sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    )  # eigs in [-2, 0].
    w, v = la.eigsh(shifted_lap, k=k, v0=d / np.linalg.norm(d))
    w += 2
    return w, v


def dispatch_for_label_usage(cls: type):
    @tf.experimental.dispatch_for_api(tf.shape, {"input": cls})
    def _shape(input, out_type: tf.DType = tf.int32, name=None):
        return input._shape(out_type=out_type, name=name)

    @tf.experimental.dispatch_for_api(tf.cast, {"x": cls})
    def _cast(x, dtype: tf.DType, name=None):
        return x._cast(dtype=dtype, name=name)

    return cls


@dispatch_for_label_usage
class PropagatedLabel(
    tf.experimental.BatchableExtensionType
):  # pylint: disable=abstract-method
    propagator: tp.Union[Matrix, tf.Tensor]
    labels: tf.Tensor

    @property
    def dtype(self):
        return self.labels.dtype

    @property
    def shape(self):
        return self.labels.shape

    def _shape(self, *args, **kwargs):
        return tf.shape(self.labels, *args, **kwargs)

    def _cast(self, *args, **kwargs):
        return PropagatedLabel(self.propagator, tf.cast(self.labels, *args, **kwargs))

    class Spec:
        propagator: tp.Any
        labels: tf.TensorSpec

        @property
        def dtype(self):
            return self.labels.dtype

        @property
        def shape(self):
            return self.labels.shape


class PropagatedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def call(self, y_true: PropagatedLabel, y_pred: tf.Tensor):
        return tf.keras.backend.sparse_categorical_crossentropy(
            y_true.labels, y_true.propagator @ y_pred
        )


class PropagatedSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def update_state(
        self, y_true: PropagatedLabel, y_pred: tf.Tensor, sample_weight=None
    ):
        super().update_state(
            y_true.labels, y_true.propagator @ y_pred, sample_weight=sample_weight
        )


class PropagatedSparseCategoricalCrossentropyMetric(
    tf.keras.metrics.SparseCategoricalCrossentropy
):
    def __init__(self, **kwargs):
        super().__init__(from_logits=False, **kwargs)

    def update_state(
        self, y_true: PropagatedLabel, y_pred: tf.Tensor, sample_weight=None
    ):
        super().update_state(
            y_true.labels, y_true.propagator @ y_pred, sample_weight=sample_weight
        )


def get_data_split(
    data: SemiSupervisedSingle,
    epsilon: float = 0.1,
    k: int = 32,
    max_iter: int = 100,
    features_transform: tp.Iterable[tp.Callable] = (),
) -> DataSplit:
    w, v = get_eig_decomp(data.adjacency, k=k)
    w = 1 / (epsilon + (1 - epsilon) * w)
    # train_prop = CompositionMatrix(
    #     [FullMatrix(w * tf.gather(v, data.train_ids)), FullMatrix(v).adjoint()]
    # )
    train_prop_adjoint = CompositionMatrix(
        [FullMatrix(v), FullMatrix(w * tf.gather(v, data.train_ids)).adjoint()]
    )
    lap = get_shifted_laplacian(data.adjacency, epsilon=epsilon)
    num_nodes = lap.shape[0]
    num_classes = int(tf.reduce_max(data.labels)) + 1
    labels_oh = tf.one_hot(data.labels, num_classes)
    labels_oh = tf.where(
        tf.expand_dims(
            tf.scatter_nd(
                tf.expand_dims(data.train_ids, 1),
                tf.ones((data.train_ids.shape[0],), dtype=tf.bool),
                (num_nodes,),
            ),
            1,
        ),
        labels_oh,
        tf.zeros_like(labels_oh),
    )
    linear_factor = (
        sparse_cg_solver(lap, max_iter=max_iter, preprocess=False) @ labels_oh
    )
    train_label = losses.LazyQuadraticCrossentropyLabelData(
        linear_factor, train_prop_adjoint
    )

    val_prop = sparse_cg_solver(
        lap, data.validation_ids, max_iter=max_iter, preprocess=True
    )
    val_label = PropagatedLabel(val_prop, tf.gather(data.labels, data.validation_ids))
    test_prop = sparse_cg_solver(lap, data.test_ids, max_iter=max_iter)
    test_label = PropagatedLabel(test_prop, tf.gather(data.labels, data.test_ids))

    features = transformed(data.node_features, features_transform)
    return DataSplit(
        train_data=((features, train_label),),
        validation_data=((features, val_label),),
        test_data=((features, test_label),),
    )


if __name__ == "__main__":
    import functools

    from graph_tf.data.single import get_data
    from graph_tf.data.transforms import row_normalize, to_format
    from graph_tf.mains.build_and_fit import build_and_fit
    from graph_tf.utils.models import mlp
    from graph_tf.utils.torch_compat import weight_decay_transformer

    k = 32

    data = get_data("cora")
    num_classes = int(tf.reduce_max(data.labels)) + 1
    split = get_data_split(
        data,
        k=k,
        features_transform=(row_normalize, functools.partial(to_format, fmt="dense")),
    )

    optimizer = tf.keras.optimizers.Adam(
        1e-2,
        gradient_transformers=[weight_decay_transformer(5e-3)],
    )

    build_and_fit(
        split,
        functools.partial(
            mlp,
            output_units=num_classes,
            hidden_units=(64,),
            dropout_rate=0.8,
        ),
        optimizer=optimizer,
        loss=losses.LazyQuadraticCategoricalCrossentropy(data.train_ids.shape[0]),
        # metrics=[
        #     PropagatedSparseCategoricalAccuracy(name="acc"),
        #     PropagatedSparseCategoricalCrossentropy(name="cross_entropy"),
        # ],
    )

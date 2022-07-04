import abc
import os
import typing as tp

import h5py
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tqdm

from graph_tf.data.transforms import transformed
from graph_tf.projects.igcn.scalable.types import (
    Features,
    FeaturesTransform,
    Transition,
)
from graph_tf.projects.igcn.scalable.utils import (
    DEFAULT_MAXITER,
    DEFAULT_TOL,
    assert_exists,
    create_transpose,
    remove_on_exception_context,
    shifted_laplacian_solver,
)
from graph_tf.utils.np_utils import block_column_generator
from graph_tf.utils.temp_utils import tempfile_context


class Sum(tf.keras.metrics.Metric):
    def __init__(self, shape: tf.TensorShape, **kwargs):
        super().__init__(**kwargs)
        self.shape = tf.TensorShape(shape)
        self.acc: tf.Variable = self.add_weight(
            "acc", shape=self.shape, dtype=self.dtype, initializer=tf.zeros
        )

    def reset_state(self):
        self.acc.assign(tf.zeros_like(self.acc))

    def update_state(self, x: tf.Tensor):
        self.acc.assign_add(x)

    def result(self) -> tf.Tensor:
        return tf.convert_to_tensor(self.acc)


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(
        self, mlp: tf.keras.Model, metrics: tp.Sequence[tf.keras.metrics.Metric]
    ):
        raise NotImplementedError

    def callback(
        self,
        metrics: tp.Sequence[tf.keras.metrics.Metric],
        prefix="val",
        validation_freq: int = 1,
        **kwargs,
    ):
        return EvaluatorCallback(
            self,
            metrics=metrics,
            prefix=prefix,
            validation_freq=validation_freq,
            **kwargs,
        )


class EvaluatorCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        evaluator: Evaluator,
        metrics: tp.Sequence[tf.keras.metrics.Metric],
        validation_freq: int = 1,
        prefix: str = "val",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.evlauator = evaluator
        self.metrics = metrics
        self.validation_freq = validation_freq
        self.prefix = prefix

    def on_epoch_end(self, epochs, logs: tp.Optional[tp.Mapping] = None):
        if epochs % self.validation_freq == 0:
            assert logs is not None
            results = self.evlauator.evaluate(self.model, metrics=self.metrics)
            logs.update({f"{self.prefix}_{k}": v for k, v in results.items()})


def eager_evaluate(
    mlp: tf.keras.Model,
    dataset: tf.data.Dataset,  # with (features, transition.T) elements
    labels: np.ndarray,
    metrics: tp.Sequence[tf.keras.metrics.Metric],
    s: tp.Optional[Sum] = None,
) -> tp.Mapping[str, tp.Any]:
    num_labels = dataset.element_spec[1].shape[1]
    num_classes = mlp.output.shape[-1]
    if s is None:
        s = Sum((num_labels, num_classes), dtype=mlp.output.dtype)
    else:
        s.reset_state()

    @tf.function
    def fn(features, transition):
        term = tf.linalg.matmul(transition, mlp(features), transpose_a=True)
        s.update_state(term)

    for features, transition in tqdm.tqdm(dataset, desc="Computing logits"):
        fn(features, transition)

    logits = s.result()

    def compute_metric(metric: tf.keras.metrics.Metric):
        metric.reset_state()
        metric.update_state(labels, logits)
        return metric.result()

    return {m.name: compute_metric(m) for m in metrics}


class EagerEvaluator(Evaluator):
    def __init__(
        self,
        dataset: tf.data.Dataset,  # with (features, transition.T) elements
        labels: tf.Tensor,
    ):
        self.dataset = dataset
        self.labels = labels
        num_labels = dataset.element_spec[1].shape[1]
        num_classes = int(tf.reduce_max(self.labels)) + 1
        self.sum = Sum((num_labels, num_classes), dtype=dataset.element_spec[1].dtype)

    def evaluate(
        self, mlp: tf.keras.Model, metrics: tp.Sequence[tf.keras.metrics.Metric]
    ) -> tp.Mapping[str, tp.Any]:
        return eager_evaluate(mlp, self.dataset, self.labels, metrics, s=self.sum)


class EagerEvaluatorManager:
    def __init__(
        self,
        root_dir: str,
        epsilon: float,
        tol: float = DEFAULT_TOL,
        maxiter: tp.Optional[int] = DEFAULT_MAXITER,
    ):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        self._base_data_path = os.path.join(root_dir, "base.h5")
        self._transpose_data_path = os.path.join(root_dir, "transpose.h5")
        self._epsilon = epsilon
        self._tol = tol
        self._maxiter = maxiter

    def transition(self, adjacency: sp.spmatrix) -> Transition:
        return shifted_laplacian_solver(
            adjacency, epsilon=self._epsilon, tol=self._tol, maxiter=self._maxiter
        )

    def create_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        attrs: tp.Optional[tp.Mapping] = None,
        block_size: int = 512,
        remove_base_data: bool = True,
    ):
        if self.has_transpose_data:
            if remove_base_data and self.has_base_data:
                self.remove_base_data()
            return
        if not self.has_base_data:
            self.create_base_data(adjacency=adjacency, ids=ids, attrs=attrs)
        self.create_transpose_data(block_size=block_size)
        if remove_base_data:
            self.remove_base_data()

    def remove_base_data(self):
        if self.has_base_data:
            os.remove(self._base_data_path)

    def create_base_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        attrs: tp.Optional[tp.Mapping] = None,
    ):
        num_labels = ids.shape[0]
        transition = self.transition(adjacency)
        num_nodes = adjacency.shape[0]

        with remove_on_exception_context(self._base_data_path):
            with h5py.File(self._base_data_path, "a") as root:
                cache = root.create_dataset(
                    "transition", shape=(num_labels, num_nodes), dtype=np.float32
                )
                for i in tqdm.trange(num_labels, desc="Computing label transitions"):
                    rhs = np.zeros((num_nodes,), dtype=np.float32)
                    rhs[ids[i]] = 1
                    cache[i] = transition(rhs)

                if attrs:
                    for k in attrs:
                        assert k not in root.attrs, k
                    root.attrs.update(**attrs)

    @property
    def has_base_data(self) -> bool:
        return os.path.exists(self._base_data_path)

    def assert_has_base_data(self):
        assert_exists(self._base_data_path)

    def create_transpose_data(self, block_size: int = 512):
        with h5py.File(self._base_data_path, "r") as src:
            with remove_on_exception_context(self._transpose_data_path):
                with h5py.File(self._transpose_data_path, "a") as dst:
                    create_transpose(
                        src["transition"], dst, "transition", block_size=block_size
                    )
                if src.attrs:
                    dst.attrs.update(src.attrs)

    @property
    def has_transpose_data(self) -> bool:
        return os.path.exists(self._transpose_data_path)

    def assert_has_transpose_data(self):
        assert_exists(self._transpose_data_path)

    def get_dataset(
        self,
        features: Features,
        batch_size: int,
        features_transform: FeaturesTransform,
    ) -> tf.data.Dataset:
        if batch_size == -1:
            batch_size = features.shape[0]
        self.assert_has_transpose_data()
        data = h5py.File(self._transpose_data_path, "r")
        transition = data["transition"]
        num_nodes, num_labels = transition.shape  # pylint: disable=no-member
        assert features.shape[0] == num_nodes
        num_features = features.shape[1]  # pylint: disable=no-member

        def gen():
            for i in range(0, num_nodes, batch_size):
                yield features[i : i + batch_size], transition[i : i + batch_size]

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec((None, num_features), dtype=tf.float32),
                tf.TensorSpec((None, num_labels), dtype=tf.float32),
            ),
        )
        length = num_nodes // batch_size + int(num_nodes % batch_size > 0)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(length))

        if features_transform:

            def map_fn(features, transition):
                return transformed(features, features_transform), transition

            dataset = dataset.map(map_fn)
        return dataset

    def evaluator(
        self,
        features: Features,
        labels: np.ndarray,
        batch_size: int,
        features_transform: FeaturesTransform,
    ) -> "EagerEvaluator":
        dataset = self.get_dataset(
            features=features,
            batch_size=batch_size,
            features_transform=features_transform,
        ).prefetch(1)
        return EagerEvaluator(dataset=dataset, labels=labels)


def lazy_evaluate(
    mlp: tf.keras.Model,
    features_dataset: tf.data.Dataset,  # features
    transition: tp.Callable[[np.ndarray], np.ndarray],
    num_nodes: int,
    ids: np.ndarray,
    labels: np.ndarray,
    metrics: tp.Sequence[tf.keras.metrics.Metric],
    *,
    block_size: int = 16,
) -> tp.Mapping[str, tp.Any]:
    num_labels = ids.shape[0]
    num_classes = mlp.output.shape[-1]
    assert labels.shape == (num_labels,)
    with tempfile_context() as path:
        with h5py.File(path, "a") as root:
            mlp_output = root.create_dataset(
                "mlp_output", shape=(num_nodes, num_classes)
            )
            i = 0
            with tqdm.tqdm(
                features_dataset, desc="Computing MLP features", total=num_nodes
            ) as prog:
                for features in features_dataset:
                    bs = features.shape[0]
                    out = mlp(features)
                    mlp_output[i : i + bs] = out
                    i += bs
                    prog.update(bs)
        with h5py.File(path, "r") as root:
            mlp_output = root["mlp_output"]
            out = np.empty((num_labels, num_classes))
            for i, rhs in tqdm.tqdm(
                enumerate(block_column_generator(mlp_output, block_size)),
                total=num_classes,
                desc="Propagating...",
            ):
                out[:, i] = transition(rhs)[ids]

    labels = tf.convert_to_tensor(labels)

    def compute_metric(metric: tf.keras.metrics.Metric):
        metric.reset_state()
        metric.update_state(labels, out)
        return metric.result()

    return {m.name: compute_metric(m) for m in metrics}


class LazyEvaluator(Evaluator):
    def __init__(
        self,
        features_dataset: tf.data.Dataset,  # features
        transition: Transition,
        num_nodes: int,
        ids: np.ndarray,
        labels: tp.Union[np.ndarray, tf.Tensor],
    ):
        self.features_dataset = features_dataset
        self.transition = transition
        self.num_nodes = num_nodes
        self.ids = ids
        self.labels = tf.convert_to_tensor(labels)

    def evaluate(
        self, mlp: tf.keras.Model, metrics: tp.Sequence[tf.keras.metrics.Metric]
    ):
        return lazy_evaluate(
            mlp,
            features_dataset=self.features_dataset,
            transition=self.transition,
            num_nodes=self.num_nodes,
            ids=self.ids,
            labels=self.labels,
            metrics=metrics,
        )


def get_features_dataset(
    features: Features, batch_size: int, features_transform: FeaturesTransform
) -> tf.data.Dataset:
    num_nodes, num_features = features.shape
    if batch_size == -1:
        batch_size = num_nodes

    def gen():
        for i in range(0, num_nodes, batch_size):
            yield features[i : i + batch_size]

    features_dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        gen, output_signature=tf.TensorSpec((None, num_features), dtype=tf.float32)
    )
    if features_transform:
        features_dataset = features_dataset.map(
            lambda f: transformed(f, features_transform)
        )
    return features_dataset

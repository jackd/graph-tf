import abc
import os
import shutil
import typing as tp

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf
import tqdm
from tflo.matrix.core import CompositionMatrix, FullMatrix

from graph_tf.data.transforms import transformed
from graph_tf.projects.igcn.scalable import jl, losses
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
from graph_tf.utils.io_utils import H5MemmapGroup
from graph_tf.utils.np_utils import block_row_generator
from graph_tf.utils.temp_utils import tempfile_context


def _update_attrs(group: h5py.File, attrs: tp.Optional[tp.Mapping]):
    if attrs:
        for k, v in attrs.items():
            if v is not None:
                group.attrs[k] = v


def sample_without_replacement(
    seed: tf.Tensor, logits_or_input_size: tp.Optional[tf.Tensor], size: int
) -> tf.Tensor:
    """
    Sampling without replacement using Gumbel trick.

    If `logits_or_input_size` is an int or rank-0 tensor, it is the input size. If it is
    rank 1, it is the logits used for sampling.

    https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    """
    if isinstance(logits_or_input_size, int) or len(logits_or_input_size.shape) == 0:
        rand = tf.random.stateless_uniform(
            (logits_or_input_size,), seed, dtype=tf.float32
        )
        _, indices = tf.nn.top_k(rand, size)
    else:
        logits = logits_or_input_size
        z = -tf.math.log(
            -tf.math.log(
                tf.random.stateless_uniform(tf.shape(logits), seed, dtype=logits.dtype)
            )
        )
        _, indices = tf.nn.top_k(logits + z, size)
    return indices


def _create_linear_data(
    path: str,
    transition: Transition,
    num_nodes: int,
    ids: np.ndarray,
    labels: np.ndarray,
    block_size: int = 512,
    attrs: tp.Optional[tp.Mapping] = None,
):
    num_classes = int(labels.max()) + 1
    with tempfile_context() as temp_path:
        with h5py.File(temp_path, "a") as tmp:
            tmp_data = tmp.create_dataset(
                "data", shape=(num_classes, num_nodes), dtype=np.float32
            )
            sol_sum = np.zeros((num_nodes,), dtype=np.float32)
            for i in tqdm.trange(num_classes, desc="Computing linear labels"):
                rhs = np.zeros((num_nodes,), dtype=np.float32)
                rhs[ids[labels == i]] = 1
                sol = transition(rhs)
                sol_sum += sol
                tmp_data[i] = sol
        with h5py.File(temp_path, "r") as tmp:
            with remove_on_exception_context(path):
                with h5py.File(path, "a") as dst:
                    create_transpose(tmp["data"], dst, "data", block_size=block_size)
                    dst.create_dataset("sum", data=sol_sum)
                    dst.attrs["num_labels"] = ids.shape[0]
                    _update_attrs(dst, attrs)


def _create_quadratic_data(
    path: str,
    transition: Transition,
    rhs: tf.data.Dataset,
    block_size: int = 512,
    attrs: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
    k = len(rhs)
    num_nodes = rhs.element_spec.shape[0]
    with tempfile_context() as temp_path:
        with h5py.File(temp_path, "a") as tmp:
            tmp_data = tmp.create_dataset(
                "data", shape=(k, num_nodes), dtype=np.float32
            )
            for i, x in enumerate(
                tqdm.tqdm(
                    rhs.as_numpy_iterator(), total=k, desc="Computing quadratic labels"
                )
            ):
                tmp_data[i] = transition(x)
        with h5py.File(path, "a") as dst:
            with h5py.File(temp_path, "r") as tmp:
                create_transpose(tmp["data"], dst, "data", block_size=block_size)
            _update_attrs(dst, attrs)


def _gather_numpy(x: np.ndarray, indices: tf.Tensor, stateful: bool = False):
    indices.shape.assert_has_rank(1)

    def numpy_fn(indices):
        return x[indices]

    out = tf.numpy_function(numpy_fn, (indices,), tf.float32, stateful=stateful)
    out.set_shape((indices.shape[0], *x.shape[1:]))
    return out


def _gather_scaled_numpy_up_to(
    x: np.ndarray, indices: tf.Tensor, k: tp.Optional[int], stateful: bool = False
):
    k_frac = np.sqrt(x.shape[1] / k)

    def numpy_fn(indices):
        return x[indices, :k] * k_frac

    indices.shape.assert_has_rank(1)

    if k is None or k == x.shape[1]:
        return _gather_numpy(x, indices, stateful=stateful)
    assert k < x.shape[1]
    out = tf.numpy_function(numpy_fn, (indices,), tf.float32, stateful=stateful)
    out.set_shape((indices.shape[0], k, *x.shape[2:]))
    return out


def sample_indices(
    seed: tf.Tensor,
    batch_size: int,
    logits: np.ndarray,
    temperature: tp.Optional[float],
) -> tf.Tensor:
    assert seed.shape == (2,), seed.shape
    num_nodes = logits.shape[0]
    if temperature is None or temperature == np.inf:
        _, indices = tf.nn.top_k(
            tf.random.stateless_uniform((num_nodes,), seed=seed), batch_size
        )
    else:
        logits = tf.convert_to_tensor(logits / temperature)
        indices = sample_without_replacement(seed, logits, batch_size)
    indices = tf.sort(indices)
    return indices


class TrainDatasetManager(abc.ABC):
    @property
    @abc.abstractmethod
    def shape(self) -> tp.Tuple[int, int]:
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        pass

    @property
    @abc.abstractmethod
    def num_labels(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def has_data(self) -> bool:
        pass

    @abc.abstractmethod
    def create_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        labels: np.ndarray,
        block_size: int = 512,
    ):
        pass

    @abc.abstractmethod
    def get_dataset(
        self,
        features: Features,
        batch_size: int,
        features_transform: FeaturesTransform,
        temperature: tp.Optional[float] = None,
        seed: int = 0,
    ) -> tf.data.Dataset:
        pass

    @abc.abstractmethod
    def loss(self) -> tf.keras.losses.Loss:
        pass

    @property
    def num_nodes(self) -> int:
        return self.shape[0]

    @property
    def num_classes(self) -> int:
        return self.shape[1]


class LinearTrainDatasetManager(TrainDatasetManager):
    def __init__(
        self,
        root_dir: str,
        epsilon: float,
        tol: float = DEFAULT_TOL,
        maxiter: tp.Optional[int] = DEFAULT_MAXITER,
    ):
        self._root_dir = root_dir
        os.makedirs(self._root_dir, exist_ok=True)
        self.tol = tol
        self.epsilon = epsilon
        self.maxiter = maxiter
        self._linear_path = os.path.join(self._root_dir, "linear.h5")

    def _transition(self, adjacency: sp.spmatrix) -> Transition:
        return shifted_laplacian_solver(
            adjacency, epsilon=self.epsilon, tol=self.tol, maxiter=self.maxiter
        )

    @property
    def has_data(self) -> bool:
        return os.path.exists(self._linear_path)

    @property
    def shape(self) -> tp.Tuple[int, int]:
        with h5py.File(self._linear_path, "r") as fp:
            return fp["data"].shape  # pylint: disable=no-member

    @property
    def dtype(self) -> np.dtype:
        with h5py.File(self._linear_path, "r") as fp:
            return fp.dtype  # pylint: disable=no-member

    @property
    def num_labels(self) -> int:
        with h5py.File(self._linear_path, "r") as fp:
            return fp.attrs["num_labels"]

    def create_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        labels: np.ndarray,
        block_size: int = 512,
    ):
        if os.path.exists(self._linear_path):
            return
        num_nodes = adjacency.shape[0]
        _create_linear_data(
            self._linear_path,
            self._transition(adjacency),
            num_nodes,
            ids,
            labels,
            block_size=block_size,
            attrs=dict(epsilon=self.epsilon, maxiter=self.maxiter, tol=self.tol),
        )
        assert_exists(self._linear_path)

    def get_dataset(
        self,
        features: Features,
        batch_size: int,
        features_transform: FeaturesTransform,
        temperature: tp.Optional[float] = None,
        seed: int = 0,
    ) -> tf.data.Dataset:
        if batch_size == -1:
            batch_size = features.shape[0]
        fp = H5MemmapGroup(h5py.File(self._linear_path, "r"))
        linear_data = fp["data"]

        def map_fn(seed: tf.Tensor):
            indices = sample_indices(seed, batch_size, fp["sum"][:], temperature)
            feats = transformed(_gather_numpy(features, indices), features_transform)
            linear = _gather_numpy(linear_data, indices)
            return feats, linear

        return (
            tf.data.Dataset.random(seed).batch(2, drop_remainder=True).map(map_fn, -1)
        )

    def loss(self):
        return losses.LinearCategoricalCrossentropy(num_labels=self.num_labels)


class HackyQuadraticTrainDatasetManager(LinearTrainDatasetManager):
    def loss(self):
        return losses.HackyQuadraticCategoricalCrossentropy(num_labels=self.num_labels)


class QuadraticTrainDatasetManager(LinearTrainDatasetManager):
    def __init__(
        self,
        root_dir: str,
        epsilon: float,
        v2: bool = False,
        tol: float = DEFAULT_TOL,
        maxiter: tp.Optional[int] = DEFAULT_MAXITER,
        quadratic_attrs: tp.Optional[tp.Mapping] = None,
    ):
        if quadratic_attrs:
            quadratic_attrs = dict(quadratic_attrs)
        else:
            quadratic_attrs = {}
        quadratic_attrs.update(epsilon=epsilon, maxiter=maxiter, tol=tol)
        self._quadratic_attrs = quadratic_attrs
        super().__init__(epsilon=epsilon, root_dir=root_dir, tol=tol, maxiter=maxiter)
        self._v2 = v2
        self._quadratic_path = os.path.join(root_dir, "quadratic.h5")

    @property
    def has_data(self) -> bool:
        return super().has_data and os.path.exists(self._quadratic_path)

    def create_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        labels: np.ndarray,
        block_size: int = 512,
    ):
        kwargs = dict(
            adjacency=adjacency,
            ids=ids,
            block_size=block_size,
        )
        super().create_data(labels=labels, **kwargs)
        self._create_quadratic_data(**kwargs)

    def _create_quadratic_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        block_size: int = 512,
    ):
        if os.path.exists(self._quadratic_path):
            return

        _create_quadratic_data(
            self._quadratic_path,
            self._transition(adjacency),
            self._create_quadratic_data_rhs(ids, adjacency.shape[0]),
            block_size=block_size,
            attrs=self._quadratic_attrs,
        )

    def _create_quadratic_data_rhs(self, ids: np.ndarray, num_nodes: int):
        def map_fn(i):
            return tf.scatter_nd([[i]], tf.ones((1,), dtype=tf.float32), (num_nodes,))

        return tf.data.Dataset.from_tensor_slices(ids).map(map_fn)

    def _get_quadratic_factor(self, fp: H5MemmapGroup, indices: tf.Tensor) -> tf.Tensor:
        return _gather_numpy(fp["data"], indices, stateful=False)

    def get_dataset(
        self,
        features: Features,
        batch_size: int,
        features_transform: FeaturesTransform,
        temperature: tp.Optional[float] = None,
        seed: int = 0,
    ) -> tf.data.Dataset:
        if batch_size == -1:
            batch_size = features.shape[0]
        fp = H5MemmapGroup(h5py.File(self._linear_path, "r"))
        linear_data = fp["data"]
        logit_data = fp["sum"][:]
        fp = H5MemmapGroup(h5py.File(self._quadratic_path))

        def map_fn(seed: tf.Tensor):
            indices = sample_indices(seed, batch_size, logit_data, temperature)
            feats = transformed(_gather_numpy(features, indices), features_transform)
            linear = _gather_numpy(linear_data, indices)
            quad = self._get_quadratic_factor(fp, indices)
            labels = losses.LazyQuadraticCrossentropyLabelData(linear, quad)
            if self._v2:
                labels = labels.to_v2()
            return (feats, labels)

        return (
            tf.data.Dataset.random(seed).batch(2, drop_remainder=True).map(map_fn, -1)
        )

    def loss(self):
        return losses.LazyQuadraticCategoricalCrossentropy(num_labels=self.num_labels)


class ProjectedQuadraticTrainDatasetManager(QuadraticTrainDatasetManager):
    def __init__(
        self,
        root_dir: str,
        epsilon: float,
        jl_eps: float = 0.1,
        seed: int = 0,
        v2: bool = False,
        batch_jl_eps: tp.Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            root_dir=root_dir,
            v2=v2,
            quadratic_attrs=dict(jl_eps=jl_eps, seed=seed),
            epsilon=epsilon,
            **kwargs,
        )
        self._batch_jl_eps = batch_jl_eps
        self._jl_eps = jl_eps
        self._seed = seed
        self._quadratic_path = os.path.join(self._root_dir, "quadratic-projected.h5")

    @classmethod
    def from_root_dir(cls, root_dir: str, v2: bool = False):
        path = os.path.join(root_dir, "quadratic-projected.h5")
        with h5py.File(path, "r") as fp:
            return cls(root_dir=root_dir, v2=v2, **fp.attrs)

    def _create_quadratic_data_rhs(self, ids: np.ndarray, num_nodes: int):
        num_labels = ids.shape[0]
        k = jl.johnson_lindenstrauss_min_dim(num_labels, eps=self._jl_eps)

        def map_fn(seed):
            R = jl.stateless_gaussian_projection_vector(
                k, num_labels, seed=seed, dtype=tf.float32
            )
            RM = tf.scatter_nd(tf.expand_dims(ids, 1), R, (num_nodes,))
            return RM

        return (
            tf.data.Dataset.random(self._seed)
            .batch(2, drop_remainder=True)
            .take(k)
            .map(map_fn)
        )

    def _get_quadratic_factor(self, fp: H5MemmapGroup, indices: tf.Tensor) -> tf.Tensor:
        if self._batch_jl_eps is not None and self._batch_jl_eps != self._jl_eps:
            batch_k = jl.johnson_lindenstrauss_min_dim(
                self.num_labels, eps=self._batch_jl_eps
            )
            data = fp["data"]
            k = data.shape[1]
            k0 = jl.johnson_lindenstrauss_min_dim(self.num_labels, eps=self._jl_eps)
            assert k0 <= k, (k0, k)
            assert batch_k < k, (batch_k, k)
            assert k <= data.shape[1], (k, data.shape[1])
            k_frac = np.sqrt(k / batch_k)

            def np_fn(indices: np.ndarray):
                cols = np.random.choice(k, size=batch_k, replace=False)
                return data[np.ix_(indices, cols)] * k_frac
                # start = np.random.uniform(low=0, high=k - batch_k, size=()).astype(
                #     np.int64
                # )
                # return data[:, start : start + batch_k][indices] * k_frac
                # return data[indices, start : start + batch_k] * k_frac

            out = tf.numpy_function(np_fn, (indices,), Tout=data.dtype, stateful=True)
            out.set_shape((indices.shape[0], batch_k))
            return out
        jl_eps = self._jl_eps
        if jl_eps > fp.attrs["jl_eps"]:
            k = jl.johnson_lindenstrauss_min_dim(self.num_labels, eps=jl_eps)
            return _gather_scaled_numpy_up_to(fp["data"], indices, k=k, stateful=False)
        return _gather_numpy(fp["data"], indices, stateful=False)


class LowRankQuadraticTrainDatasetManager(LinearTrainDatasetManager):
    def __init__(
        self,
        root_dir: str,
        epsilon: float,
        k: int = 32,
        tol: float = DEFAULT_TOL,
        maxiter: tp.Optional[int] = DEFAULT_MAXITER,
    ):
        super().__init__(
            root_dir=root_dir,
            epsilon=epsilon,
            tol=tol,
            maxiter=maxiter,
        )
        self._k = k
        self._quadratic_path = os.path.join(root_dir, "low-rank-quadratic.h5")

    @property
    def has_data(self) -> bool:
        return super().has_data and os.path.exists(self._quadratic_path)

    def create_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        labels: np.ndarray,
        block_size: int = 512,
    ):
        if self.has_data:
            return
        kwargs = dict(
            adjacency=adjacency,
            ids=ids,
            block_size=block_size,
        )
        super().create_data(labels=labels, **kwargs)
        self._create_quadratic_data(**kwargs)

    def _create_quadratic_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        block_size: int = 512,
    ):
        del block_size
        adjacency = adjacency.tocoo(copy=False)
        n = adjacency.shape[0]
        d = np.squeeze(np.asarray(adjacency.sum(axis=1)), axis=1)
        d_norm = d.sum()
        d = np.sqrt(d)
        data = 1 / (d[adjacency.row] * d[adjacency.col])
        L = -(
            sp.eye(n)
            + sp.coo_matrix((data, (adjacency.row, adjacency.col)), shape=(n, n))
        )  # L - 2I, eigenvalues in [-2, 0]
        w, v = la.eigsh(L, v0=d / d_norm, k=self._k, which="LM", tol=1e-2, maxiter=1000)
        w += 2
        w = w.astype(np.float32)
        v = v.astype(np.float32)
        with remove_on_exception_context(self._quadratic_path):
            with h5py.File(self._quadratic_path, "a") as fp:
                fp.create_dataset("w", data=w)
                fp.create_dataset("v", data=v)
                fp.create_dataset("ids", data=ids)

    def get_dataset(
        self,
        features: Features,
        batch_size: int,
        features_transform: FeaturesTransform,
        temperature: tp.Optional[float] = None,
        seed: int = 0,
    ) -> tf.data.Dataset:
        if batch_size == -1:
            batch_size = features.shape[0]
        fp = H5MemmapGroup(h5py.File(self._linear_path, "r"))
        linear_data = fp["data"]
        logit_data = fp["sum"][:]
        fp = H5MemmapGroup(h5py.File(self._quadratic_path))
        v = fp["v"]
        v_labels = tf.convert_to_tensor(
            v[fp["ids"], : self._k]
            / (self.epsilon + (1 - self.epsilon) * fp["w"][: self._k]),
            tf.float32,
        )  # [L, K]

        # if batch_size == features.shape[0]:
        #     feats = transformed(tf.convert_to_tensor(features), features_transform)
        #     linear = tf.convert_to_tensor(linear_data)
        #     v_features = tf.convert_to_tensor(v[:, : self._k])
        #     quad = CompositionMatrix(
        #         (FullMatrix(v_features), FullMatrix(v_labels).adjoint())
        #     )  # [B, L]
        #     labels = losses.LazyQuadraticCrossentropyLabelData(linear, quad)
        #     dataset = tf.data.Dataset.from_tensors((feats, labels))
        #     return dataset.repeat()

        # assert batch_size < features.shape[0], (batch_size, features.shape[0])

        def map_fn(seed: tf.Tensor):
            indices = sample_indices(seed, batch_size, logit_data, temperature)
            feats = transformed(_gather_numpy(features, indices), features_transform)
            linear = _gather_numpy(linear_data, indices)
            if self._k < v.shape[1]:
                v_features = _gather_scaled_numpy_up_to(v, indices, self._k)
            else:
                v_features = _gather_numpy(v, indices)  # [B, K]
            quad = CompositionMatrix(
                (FullMatrix(v_features), FullMatrix(v_labels).adjoint())
            )  # [B, L]
            labels = losses.LazyQuadraticCrossentropyLabelData(linear, quad)
            return (feats, labels)

        return (
            tf.data.Dataset.random(seed).batch(2, drop_remainder=True).map(map_fn, -1)
        )

    def loss(self):
        return losses.LazyQuadraticCategoricalCrossentropy(num_labels=self.num_labels)


class DummyTrainDatasetManager(TrainDatasetManager):
    def __init__(self, root_dir: str, jl_eps: float, seed: int = 0):
        self._jl_eps = jl_eps
        self._root_dir = root_dir
        self._seed = seed
        self._data_dir = os.path.join(self._root_dir, "dummy")
        self._num_labels = self._shape = self._dtype = self._k = None

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def has_data(self) -> bool:
        return self._num_labels is not None

    def create_data(
        self,
        adjacency: sp.spmatrix,
        ids: np.ndarray,
        labels: np.ndarray,
        block_size: int = 512,
    ):
        num_nodes = adjacency.shape[0]
        self._num_labels = ids.shape[0]
        num_classes = int(labels.max()) + 1
        self._shape = (num_nodes, num_classes)
        self._dtype = np.float32
        self._k = jl.johnson_lindenstrauss_min_dim(self._num_labels, eps=self._jl_eps)
        if os.path.exists(self._data_dir):
            return
        with remove_on_exception_context(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)

            def gen():
                rng = np.random.default_rng(self._seed)
                for _ in tqdm.trange(num_nodes, desc="Creating dummy data..."):
                    linear = rng.uniform(size=(num_classes,)).astype(np.float32)
                    quadratic = rng.uniform(size=(self._k,)).astype(np.float32)
                    yield linear, quadratic

            dataset = tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    tf.TensorSpec((num_classes,), tf.float32),
                    tf.TensorSpec((self._k,), tf.float32),
                ),
            )
            tf.data.experimental.save(dataset, self._data_dir)

    def get_dataset(
        self,
        features: Features,
        batch_size: int,
        features_transform: FeaturesTransform,
        temperature: tp.Optional[float] = None,
        seed: int = 0,
    ) -> tf.data.Dataset:
        if batch_size == -1:
            batch_size = features.shape[0]
        features_dir = os.path.join(self._data_dir, "features")
        if os.path.exists(features_dir):
            shutil.rmtree(features_dir)
        os.makedirs(features_dir)

        def gen():
            return tqdm.tqdm(
                block_row_generator(features),
                total=features.shape[0],
                desc="Creating features dataset",
            )

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=tf.TensorSpec((features.shape[1],), dtype=features.dtype),
        )
        with remove_on_exception_context(features_dir):
            tf.data.experimental.save(dataset, features_dir)

        features_ds = tf.data.experimental.load(features_dir)
        label_ds = tf.data.experimental.load(self._data_dir)
        prob = 0.5
        bs = int(batch_size / prob)

        def map_fn(seed, features, labels):
            labels = losses.LazyQuadraticCrossentropyLabelData(*labels)
            p = tf.random.stateless_uniform((bs,), seed)
            _, indices = tf.nn.top_k(p, batch_size)
            return tf.nest.map_structure(
                lambda x: tf.gather(x, indices),
                (features, labels),
                expand_composites=True,
            )

        return tf.data.Dataset.zip(
            (
                tf.data.Dataset.random(seed).batch(2),
                features_ds.repeat().batch(bs),
                label_ds.repeat().batch(bs),
            )
        ).map(map_fn, -1)
        # return (
        #     tf.data.Dataset.zip(
        #         (
        #             tf.data.Dataset.random(seed)
        #             .batch(2)
        #             .map(lambda s: tf.random.stateless_uniform((), seed=s) < prob),
        #             features_ds,
        #             label_ds.map(losses.LazyQuadraticCrossentropyLabelData),
        #         )
        #     )
        #     .repeat()
        #     .filter(lambda keep, features, labels: keep)
        #     .map(lambda k, features, labels: (features, labels))
        #     .batch(batch_size)
        # )
        # num_features = features.shape[1]

        # def map_fn(seed: tf.Tensor):
        #     seeds = tf.unstack(
        #         tf.random.experimental.stateless_split(seed, num=3), axis=0
        #     )
        #     features = tf.random.stateless_normal((num_features,), seed=seeds.pop())
        #     linear = tf.random.stateless_uniform(
        #         (self.num_classes,), seed=seeds.pop()
        #     )
        #     quad = tf.random.stateless_uniform((self._k,), seed=seeds.pop())
        #     assert not seeds
        #     return features, losses.LazyQuadraticCrossentropyLabelData(linear, quad)

        # return tf.data.Dataset.random(seed).batch(2).map(map_fn, -1).batch(batch_size)

    def loss(self) -> tf.keras.losses.Loss:
        return losses.LazyQuadraticCategoricalCrossentropy(num_labels=self._num_labels)

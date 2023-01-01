import functools
import os
import typing as tp

import gin
import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf
import tqdm

from graph_tf.data.data_types import DataSplit, TransitiveData
from graph_tf.utils.io_utils import H5MemmapGroup
from graph_tf.utils.np_utils import write_block_rows

register = functools.partial(gin.register, module="gtf.data.cache")


def column_generator(x: np.ndarray):
    return (x[:, i] for i in range(x.shape[1]))


def page_rank_propagate(
    adj: sp.spmatrix,
    x: np.ndarray,
    epsilon: float,
    symmetric: bool = True,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> tp.Iterable[np.ndarray]:
    assert x.ndim == 2

    if epsilon == 1:
        return column_generator(x)
    else:
        if not symmetric:
            raise NotImplementedError()
        d = np.squeeze(np.asarray(adj.sum(1)), axis=1)
        x0 = np.sqrt(d)
        d_rsqrt = 1 / x0
        adj = adj.tocoo()
        data = adj.data * d_rsqrt[adj.row] * d_rsqrt[adj.col]
        adj = sp.coo_matrix((data, (adj.row, adj.col)), shape=adj.shape)
        shifted_lap = sp.eye(adj.shape[0], dtype=adj.dtype) + adj * (epsilon - 1)

        def solve(xi: np.ndarray) -> np.ndarray:
            x, info = la.cg(shifted_lap, xi, x0=x0, tol=tol, maxiter=max_iter)
            del info
            return x

        return (solve(xi) for xi in column_generator(x))


class NumpyClassificationData(tp.NamedTuple):
    features: tp.Sequence[np.ndarray]
    labels: np.ndarray


def _get_page_rank_cache(
    path: str,
    epsilon: tp.Union[float, tp.Iterable[float]],
    data_fn: tp.Callable[[], TransitiveData],
    max_iter: int = 100,
    tol: float = 1e-2,
    show_progress: bool = True,
    memmap: bool = True,
    splits: tp.Sequence[str] = ("train", "validation", "test"),
) -> tp.Sequence[NumpyClassificationData]:
    epsilon = (epsilon,) if isinstance(epsilon, (int, float)) else tuple(epsilon)

    def key(eps: float):
        if eps == 1.0:
            return "base"
        return f"page-rank-{eps:.1e}-{tol:.1e}-{max_iter}"

    if os.path.exists(path):
        with h5py.File(path, "r") as root:
            has_features = all(key(eps) in root for eps in epsilon)
            has_labels = all(split in root["labels"] for split in splits)
            has_data = has_features and has_labels
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        has_data = False
    if not has_data:
        data = data_fn()
        # create labels data
        assert isinstance(data, TransitiveData)
        ids_list = [getattr(data, f"{split}_ids") for split in splits]
        for split, ids in zip(splits, ids_list):
            if ids is None:
                raise ValueError(f"Data has no {split} split")

        # create features data
        features = data.node_features
        num_features = features.shape[1]
        with h5py.File(path, "a") as fp:
            # copy labels
            if "labels" not in fp:
                labels_group = fp.create_group("labels")
                try:
                    src_labels = np.array(data.labels)
                    for split, ids in zip(splits, ids_list):
                        labels_group.create_dataset(split, data=src_labels[ids])
                except (KeyboardInterrupt, Exception):
                    del fp["labels"]
                    raise
            # copy features
            for eps in epsilon:
                k = key(eps)
                if k in fp:
                    continue
                try:
                    dst_group = fp.create_group(k)
                    dst_list = [
                        dst_group.create_dataset(
                            split,
                            shape=(ids.shape[0], num_features),
                            dtype=features.dtype,
                        )
                        for split, ids in zip(splits, ids_list)
                    ]
                    if eps == 1.0:
                        # extra branch to avoid loading data.adjacency if lazily loaded
                        for split, dst, ids in zip(splits, dst_list, ids_list):
                            if show_progress:
                                ids = tqdm.tqdm(ids, desc=f"Copying {split} features")
                            feats_it = (features[id] for id in ids)
                            write_block_rows(feats_it, dst, block_size=512)

                    else:
                        src = page_rank_propagate(
                            data.adjacency,
                            features,
                            epsilon=eps,
                            symmetric=True,
                            tol=tol,
                            max_iter=max_iter,
                        )
                        if show_progress:
                            src = tqdm.tqdm(
                                src,
                                total=features.shape[1],
                                desc=f"Computing page rank, epsilon={eps}",
                            )
                            for i, col in enumerate(src):
                                for dst, ids in zip(dst_list, ids_list):
                                    dst[:, i] = col[ids]
                except (KeyboardInterrupt, Exception):
                    if k in fp:
                        del fp[k]
                    raise

    root = h5py.File(path, "r")
    if memmap:
        root = H5MemmapGroup(root)
    labels_list = [np.array(root["labels"][split]) for split in splits]
    features_list = [[] for _ in splits]
    for eps in epsilon:
        k = key(eps)
        dst_group = root[k]
        for split, fl in zip(splits, features_list):
            fl.append(dst_group[split])
    return tuple(
        NumpyClassificationData(fl, ll) for fl, ll in zip(features_list, labels_list)
    )


def _cached_features_to_dataset(
    features: tp.Sequence[np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    shuffle_seed: tp.Optional[int] = None,
    prefetch_buffer: int = 1,
    weighted: bool = False,
) -> tf.data.Dataset:
    assert features
    size = labels.shape[0]
    assert all(f.shape[0] == size for f in features), (
        tuple(f.shape[0] for f in features),
        size,
    )

    def np_fn(ids):
        return np.concatenate([f[ids] for f in features], axis=-1), labels[ids]

    def tf_fn(ids: tf.Tensor) -> tf.Tensor:
        feats, labs = tf.numpy_function(
            np_fn, (ids,), (tf.float32, labels.dtype), stateful=False
        )
        feats.set_shape((*ids.shape, sum(f.shape[1] for f in features)))
        labs.set_shape(ids.shape)
        if weighted:
            weights = tf.fill(tf.shape(weights), 1 / size)
            return feats, labs, weights
        return feats, labs

    dataset = tf.data.Dataset.range(size)
    if shuffle_seed is not None:
        dataset = dataset.shuffle(len(dataset), seed=shuffle_seed)
    return dataset.batch(batch_size).map(tf_fn).prefetch(prefetch_buffer)


@register
def cached_page_rank_splits(
    cache_path: str,
    data_fn: tp.Callable[[], TransitiveData],
    epsilon: tp.Union[float, tp.Iterable[float]],
    max_iter: int = 1000,
    tol: float = 1e-2,
    batch_size: int = 256,
    shuffle_seed: int = 0,
    show_progress: bool = True,
    prefetch_buffer: int = 1,
    in_memory: bool = False,
    weighted: bool = False,
    splits=("train", "validation", "test"),
) -> DataSplit:
    split_data = _get_page_rank_cache(
        cache_path,
        epsilon=epsilon,
        tol=tol,
        max_iter=max_iter,
        show_progress=show_progress,
        data_fn=data_fn,
        splits=splits,
    )
    split_data_dict = dict(zip(splits, split_data))

    def to_dataset(
        class_data: tp.Optional[NumpyClassificationData],
        training: bool = False,
    ):
        if class_data is None:
            return None
        features, labels = class_data

        if in_memory:
            features = np.concatenate(features, axis=1)
            if weighted:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (
                        features,
                        labels,
                        tf.fill(labels.shape, 1 / labels.shape[0]),
                    )
                )
            else:
                dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            if training and shuffle_seed is not None:
                dataset = dataset.shuffle(len(dataset), shuffle_seed)
            return dataset.batch(batch_size).prefetch(prefetch_buffer)

        return _cached_features_to_dataset(
            features,
            labels,
            batch_size,
            shuffle_seed if training else None,
            prefetch_buffer,
            weighted=weighted,
        )

    return DataSplit(
        to_dataset(split_data_dict.get("train"), training=True),
        to_dataset(split_data_dict.get("validation")),
        to_dataset(split_data_dict.get("test")),
    )


if __name__ == "__main__":
    import sys

    from graph_tf.data.transitive import cached_ogbn_papers100m
    from graph_tf.utils.os_utils import get_dir

    epsilon = [float(arg) for arg in sys.argv[1:]]
    if len(epsilon) == 0:
        print("No epsilon provided. Usage: `python cache.py *eps")
        exit()
    cache_path = os.path.join(
        get_dir(None, "GTF_DATA_DIR", "~/graph-tf-data"),
        "page-rank",
        "ogbn-papers100m.h5",
    )
    data_fn = cached_ogbn_papers100m
    split = cached_page_rank_splits(
        cache_path=cache_path,
        data_fn=data_fn,
        epsilon=epsilon,
        max_iter=1000,
        tol=1e-2,
        batch_size=64,
        in_memory=False,
    )
    for features, labels in tqdm.tqdm(split.train_data, desc="Iterating train_ds"):
        pass

    for features, labels in tqdm.tqdm(
        split.validation_data, desc="Iterating validation_ds"
    ):
        pass

    for features, labels in tqdm.tqdm(split.test_data, desc="Iterating test_ds"):
        pass

    print("Iteration complete")

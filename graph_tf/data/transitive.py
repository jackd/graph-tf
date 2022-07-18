import functools
import os
import shutil
import typing as tp
import zipfile

import gin
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tqdm
import wget

from graph_tf.data.data_types import TransitiveData
from graph_tf.utils.io_utils import mmap_h5
from graph_tf.utils.np_utils import ArrayStream, write_block_rows
from graph_tf.utils.os_utils import get_dir

register = functools.partial(gin.register, module="gtf.data.transitive")


def _create_ogbn_papers100m_cache(cache_path: str, ogb_dir: str):

    assert not os.path.exists(cache_path)
    data_dir = os.path.join(ogb_dir, "ogbn_papers100M")

    url = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
    split_paths = tuple(
        os.path.join(data_dir, "split", "time", f"{split}.csv.gz")
        for split in ("train", "valid", "test")
    )

    def has_raw_data():
        raw_paths = (
            (os.path.join(data_dir, "raw", fn)) for fn in ("data.npz", "node-label.npz")
        )
        has_raw_data = all(os.path.exists(rp) for rp in raw_paths)
        has_split_data = all(os.path.exists(sp) for sp in split_paths)
        return has_raw_data and has_split_data

    if not has_raw_data():
        zip_path = os.path.join(ogb_dir, "papers100M-bin.zip")
        if not os.path.exists(zip_path):
            # download
            print("Downloading papers100M data...")
            wget.download(url, zip_path)
        assert os.path.exists(zip_path)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(ogb_dir)
        shutil.move(
            os.path.join(ogb_dir, "papers100M-bin"),
            data_dir,
        )
        assert has_raw_data()
        if os.path.exists(zip_path):
            os.remove(zip_path)

    try:
        with h5py.File(cache_path, "a") as root:
            # copy split indices
            for split, path in zip(("train", "validation", "test"), split_paths):
                root.create_dataset(
                    f"{split}_ids",
                    data=pd.read_csv(path, compression="gzip", header=None)
                    .values.T[0]
                    .astype(np.int64),
                )
            labels = np.load(os.path.join(data_dir, "raw", "node-label.npz"))
            labels = labels["node_label"]
            labels[np.isnan(labels)] = -1
            labels = labels.astype(np.int64)
            if labels.ndim == 2:
                labels = np.squeeze(labels, axis=1)
            root.create_dataset("labels", data=labels)
            root.attrs["num_classes"] = labels.max() + 1

            data = np.load(os.path.join(data_dir, "raw", "data.npz"))
            print("Copying adjacency...")
            adj = root.create_group("adjacency")
            src = ArrayStream.from_npz(data, "edge_index")
            it = iter(src.as_iterable())
            print("  Copying row indices...")
            adj.create_dataset("row", data=next(it))
            print("  Copying col indices...")
            adj.create_dataset("col", data=next(it))
            # pylint: disable=no-member
            assert adj["row"].shape == adj["col"].shape
            assert adj["row"].shape != (2,)
            root.attrs["num_edges"] = adj["row"].shape
            # pylint: enable=no-member

            # copy node features
            src = ArrayStream.from_npz(data, "node_feat")
            root.attrs["num_nodes"], root.attrs["num_features"] = src.shape
            dst = root.create_dataset("node_features", shape=src.shape, dtype=src.dtype)
            write_block_rows(
                tqdm.tqdm(src, desc="Writing features"), dst, block_size=8192
            )
    except (KeyboardInterrupt, Exception):
        if os.path.exists(cache_path):
            os.remove(cache_path)
        raise


class Papers100MData(TransitiveData):
    def __init__(self, root: h5py.Group, symmetric: bool):
        self.root = root
        self.symmetric = symmetric

    @property
    @functools.lru_cache(1)
    def adjacency(self) -> sp.coo_matrix:
        print("Lazily loading adjacency...")
        root = self.root
        n = root.attrs["num_nodes"]
        adj = root["adjacency"]
        row = adj["row"][:]
        col = adj["col"][:]
        if self.symmetric:
            print("  Making symmetric...")
            valid = row < col
            row = row[valid]
            col = col[valid]
            row, col = np.concatenate((row, col)), np.concatenate((col, row))
            i1d: np.ndarray = np.ravel_multi_index((row, col), (n, n))
            i1d.sort()  # pylint: disable=no-member
            row, col = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
                i1d, (n, n)
            )

        adj = sp.coo_matrix(
            (np.ones((row.shape[0],), dtype=np.float32), (row, col)), shape=(n, n)
        )
        print("  Done")
        return adj

    @property
    def node_features(self):
        return mmap_h5(self.root["node_features"])

    @property
    def labels(self):
        return self.root["labels"][:]

    @property
    def train_ids(self):
        return self.root["train_ids"][:]

    @property
    def validation_ids(self):
        return self.root["validation_ids"][:]

    @property
    def test_ids(self):
        return self.root["test_ids"][:]


@register
def cached_ogbn_papers100m(
    gtf_data_dir: tp.Optional[str] = None,
    ogb_dir: tp.Optional[str] = None,
    symmetric: bool = True,
) -> TransitiveData:

    gtf_data_dir = get_dir(gtf_data_dir, "GTF_DATA_DIR", "~/graph-tf-data")
    ogb_dir = get_dir(ogb_dir, "OGB_DATA", "~/ogb")
    cache_path = os.path.join(gtf_data_dir, "base", "ogbn-papers100m.h5")
    if not os.path.exists(cache_path):
        _create_ogbn_papers100m_cache(cache_path, ogb_dir)

    root = h5py.File(cache_path, "r")
    return Papers100MData(root, symmetric=symmetric)

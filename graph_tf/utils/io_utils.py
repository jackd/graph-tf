import json
import os
import typing as tp

import h5py
import numpy as np
import tensorflow as tf
import tqdm

# import zarr
Writer = tp.Callable[[str, tf.data.Dataset], None]
Reader = tp.Callable[[str], np.ndarray]


def _dataset_meta(dataset: tf.data.Dataset) -> tp.Tuple[tp.Tuple[int, ...], np.dtype]:
    spec = dataset.element_spec
    assert isinstance(spec, tf.TensorSpec)
    total = len(dataset)
    shape = (total, *spec.shape)
    dtype = spec.dtype.as_numpy_dtype
    return shape, dtype


def _write(sink: np.ndarray, dataset: tf.data.Dataset, verbose: bool, desc: str):
    dataset = dataset.as_numpy_iterator()
    if verbose:
        dataset = tqdm.tqdm(dataset, desc=desc, total=sink.shape[0])
    for i, el in enumerate(dataset):
        sink[i] = el


##########################
# h5py
##########################


def _h5_path(path: str):
    if not path.endswith(".h5"):
        path = f"{path}.h5"
    return path


def write_hdf5(path: str, dataset: tf.data.Dataset, verbose: bool = True):
    path = _h5_path(path)
    shape, dtype = _dataset_meta(dataset)
    with h5py.File(path, "a") as fp:
        sink = fp.create_dataset("data", shape=shape, dtype=dtype)
        _write(sink, dataset, verbose=verbose, desc=f"Writing hdf5 to {path}")


def read_hdf5(path: str) -> h5py.Dataset:
    return h5py.File(_h5_path(path), "r")["data"]


##########################
# np.memmap
##########################


def _data_path(path: str):
    return os.path.join(path, "data.dat")


def _meta_path(path: str):
    return os.path.join(path, "meta.json")


def write_memmap(path: str, dataset: tf.data.Dataset, verbose: bool = True):
    shape, dtype = _dataset_meta(dataset)

    sink = np.memmap(_data_path(path), dtype=dtype, shape=shape, mode="w+")
    _write(sink, dataset, verbose=verbose, desc=f"Writing memmap to {path}")
    sink.flush()
    with open(_meta_path(path), "w", encoding="utf-8") as fp:
        json.dump(dict(dtype=str(dtype)[14:-2], shape=list(shape)), fp)


def read_memmap(path: str) -> np.memmap:
    with open(_meta_path(path), "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    shape = tuple(meta["shape"])
    dtype = getattr(np, meta["dtype"])
    return np.memmap(_data_path(path), mode="r", dtype=dtype, shape=shape)


##########################
# np.save / np.load
##########################
def _numpy_path(path: str):
    return os.path.join(path, "data.npy")


def write_numpy(path: str, dataset: tf.data.Dataset, verbose: bool = True):
    shape, dtype = _dataset_meta(dataset)
    sink = np.empty(shape=shape, dtype=dtype)
    _write(sink, dataset, verbose=verbose, desc="Writing to memory")
    np.save(_numpy_path(path), sink)


def read_numpy(path: str):
    return np.load(_numpy_path(path))


##########################
# zarr
##########################


# class _ZarrArray:
#     def __init__(self, x):
#         self.x = x

#     @property
#     def shape(self):
#         return self.x.shape

#     @property
#     def dtype(self):
#         return self.x.dtype

#     def __getitem__(self, i):
#         mask = np.zeros(self.x.shape, dtype=np.bool)
#         mask[i] = True
#         return self.x.get_mask_selection(mask)
#         # return self.x.get_coordinate_selection((i, [slice(None)]))


# def _zarr_path(path: str):
#     return os.path.join(path, "data.zarr")


# def write_zarr(path: str, dataset: tf.data.Dataset, verbose: bool = True):
#     shape, dtype = _dataset_meta(dataset)
#     sink = zarr.open(_zarr_path(path), mode="w", shape=shape, dtype=dtype)
#     _write(sink, dataset, verbose=verbose, desc="Writing zarr")


# def read_zarr(path: str):
#     return _ZarrArray(zarr.open(_zarr_path(path), mode="r"))


def mmap_h5(ds: h5py.Dataset) -> np.memmap:
    # from https://gist.github.com/rossant/7b4704e8caeb8f173084
    # We get the dataset address in the HDF5 field.
    offset = ds.id.get_offset()
    assert offset > 0
    # We ensure we have a non-compressed contiguous array.
    assert ds.chunks is None
    assert ds.compression is None
    mode = ds.file.mode
    arr = np.memmap(
        ds.file.filename,
        mode="w+" if mode == "w" else mode,
        shape=ds.shape,
        offset=offset,
        dtype=ds.dtype,
    )
    return arr


class H5MemmapGroup(tp.Mapping[str, tp.Union["H5MemmapGroup", np.memmap]]):
    """
    Wrapper around `h5py.Group` that wraps any returned datasets in memmaps.

    This gives better fancy indexing performance.

    See https://gist.github.com/rossant/7b4704e8caeb8f173084
    """

    def __init__(self, group: h5py.Group):
        self.root = group

    @property
    def attrs(self):
        return self.root.attrs

    def create_group(self, name: str) -> "H5MemmapGroup":
        return H5MemmapGroup(self.root.create_group(name))

    def require_group(self, name: str) -> "H5MemmapGroup":
        return H5MemmapGroup(self.root.require_group(name))

    def create_dataset(
        self,
        name: str,
        shape: tp.Optional[tp.Sequence[int]] = None,
        dtype: tp.Optional[np.dtype] = None,
        data: tp.Optional[np.ndarray] = None,
    ) -> np.memmap:
        return mmap_h5(
            self.root.create_dataset(name=name, shape=shape, dtype=dtype, data=data)
        )

    def require_dataset(
        self,
        name: str,
        shape: tp.Optional[tp.Sequence[int]] = None,
        dtype: tp.Optional[np.dtype] = None,
        data: tp.Optional[np.ndarray] = None,
    ) -> np.memmap:
        return mmap_h5(
            self.root.require_dataset(name=name, shape=shape, dtype=dtype, data=data)
        )

    def __getitem__(self, key: str):
        value = self.root[key]
        if isinstance(value, h5py.Dataset):
            return mmap_h5(value)
        assert isinstance(value, h5py.Group)
        return H5MemmapGroup(value)

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)

    def __contains__(self, key):
        return key in self.root

    def __enter__(self):
        if isinstance(self.root, h5py.File):
            self.root.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        if isinstance(self.root, h5py.File):
            self.root.__exit__(*args, **kwargs)


class MemmapStore(tp.Mapping[str, tp.Union["MemmapStore", np.memmap]]):
    def __init__(self, root_dir: str, mode: str = "w+"):
        self.root_dir = root_dir
        assert os.path.exists(self.root_dir)
        self.mode = mode
        assert mode in ("r", "r+", "w+")

    def _create_group(self, name: str, exist_ok: bool) -> "MemmapStore":
        data_path = self._data_path(name)
        if os.path.isfile(data_path):
            raise ValueError(f"Cannot create group: data alraedy exists at {data_path}")
        path = os.path.join(self.root_dir, name)
        os.makedirs(path, exist_ok=exist_ok)
        return MemmapStore(path)

    def create_group(self, name: str) -> "MemmapStore":
        return self._create_group(name, exist_ok=False)

    def require_group(self, name: str) -> "MemmapStore":
        return self._create_group(name, exist_ok=True)

    def _data_path(self, group_path: str):
        return os.path.join(group_path, "data.dat")

    def _meta_path(self, group_path: str):
        return os.path.join(group_path, "meta.json")

    def _group_path(self, name: str):
        return os.path.join(self.root_dir, name)

    def create_dataset(
        self,
        name: str,
        shape: tp.Optional[tp.Sequence[int]] = None,
        dtype: tp.Optional[np.dtype] = None,
        data: tp.Optional[np.ndarray] = None,
    ) -> np.memmap:
        if self.mode in ("r", "r+"):
            raise ValueError(
                f"Cannot create dataset in read mode (mode=='{self.mode}')"
            )
        group_path = self._group_path(name)
        if os.path.isdir(group_path):
            raise ValueError(f"Data/group already exists at {group_path}")
        os.makedirs(group_path, exist_ok=False)
        data_path = self._data_path(group_path)
        meta_path = self._data_path(group_path)

        if data is None:
            assert shape is not None
            assert dtype is not None
            out = np.memmap(data_path, dtype=dtype, shape=shape, mode="w+")
        else:
            assert shape is None or tuple(shape) == data.shape, (shape, data.shape)
            assert dtype is None or dtype == data.dtype, (dtype, data.dtype)
            out = np.memmap(
                data_path, shape=data.shape, dtype=data.dtype, mode=self.mode
            )
            out[:] = data
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(dict(dtype=str(dtype)[14:-2], shape=list(shape)), fp)
        return out

    def _get_memmap(self, group_path: str):
        data_path = self._data_path(group_path)
        if not os.path.isfile(data_path):
            return None
        meta_path = self._meta_path(group_path)
        assert os.path.isfile(meta_path), meta_path
        with open(meta_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
            return np.memmap(data_path, mode=self.mode, **meta)

    def __getitem__(self, key: str) -> tp.Union["MemmapStore", np.memmap]:
        group_path = self._group_path(key)
        if not os.path.isdir(group_path):
            raise KeyError(f"No directory at {group_path}")
        memmap = self._get_memmap(group_path)
        if memmap is None:
            return MemmapStore(group_path)
        else:
            return memmap

    def __contains__(self, key: str):
        return os.path.isdir(self._group_path(key))

    def keys(self):
        return os.listdir(self.root_dir)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(os.listdir(self.root_dir))

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

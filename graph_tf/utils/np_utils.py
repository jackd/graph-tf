import contextlib
import pathlib
import typing as tp

import numpy as np


@contextlib.contextmanager
def random_seed_context(seed: int):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


class ArrayStream:
    def __init__(self, fp):
        version = np.lib.format.read_magic(fp)
        np.lib.format._check_version(version)
        self._shape, fortran_order, self._dtype = np.lib.format._read_array_header(
            fp, version
        )

        if fortran_order:
            raise NotImplementedError()
        if len(self._shape) == 0:
            raise ValueError("Cannot stream 0d array.")
        self.element_count = np.multiply.reduce(self._shape[1:], dtype=np.int64)
        self.read_size = int(self.element_count * self.dtype.itemsize)
        self.count = self._shape[0] * self.element_count
        if self.dtype.hasobject:
            raise NotImplementedError()
        if np.compat.isfileobj(fp):
            raise NotImplementedError()
        self.fp = fp
        self._consumed = False

    def __len__(self) -> int:
        return self._shape[0]

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        return self._shape

    @property
    def element_shape(self) -> tp.Tuple[int, ...]:
        return self._shape[1:]

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def consumed(self) -> bool:
        return self._consumed

    def as_iterable(self) -> tp.Iterable[np.ndarray]:
        if self._consumed:
            raise ValueError("ArrayStream consumed.")
        element_shape = self.element_shape
        for _ in range(0, self.count, self.element_count):
            data = np.lib.format._read_bytes(  # pylint: disable=protected-access
                self.fp, self.read_size, "array data"
            )
            array = np.frombuffer(data, dtype=self.dtype, count=self.element_count)
            yield array.reshape(element_shape)
        self._consumed = True

    def __iter__(self):
        return iter(self.as_iterable())

    @classmethod
    def from_npz(
        cls, npz: tp.Union[str, pathlib.Path, np.lib.npyio.NpzFile], key: str
    ) -> "ArrayStream":
        if isinstance(npz, (str, pathlib.Path)):
            npz = np.load(npz)
        assert isinstance(npz, np.lib.npyio.NpzFile), npz
        assert key in npz.files
        fp = npz.zip.open(f"{key}.npy")
        return ArrayStream(fp)


def block_column_generator(x: np.ndarray, block_size: int = 512):
    for i in range(x.shape[1]):
        j = i % block_size
        if j == 0:
            block = x[:, i : i + block_size]
        yield block[:, j]


def block_row_generator(x: np.ndarray, block_size: int = 512):
    for i in range(x.shape[0]):
        j = i % block_size
        if j == 0:
            block = x[i : i + block_size]
        yield block[j]


def write_block_columns(
    columns: tp.Iterable[np.ndarray], dst: np.ndarray, block_size: int
):
    if block_size == 1:
        count = -1
        for count, el in enumerate(columns):
            dst[:, count] = el
        count += 1
    else:
        assert block_size > 0, block_size
        count = 0
        buffer = []
        for column in columns:
            buffer.append(column)
            if len(buffer) == block_size:
                dst[:, count : count + block_size] = np.stack(buffer, axis=1)
                buffer = []
                count += block_size
        if buffer:
            block_size = len(buffer)
            dst[:, count : count + block_size] = np.stack(buffer, axis=1)
            count += block_size
    assert count == dst.shape[0], (count, dst.shape[0])


def write_block_rows(rows: tp.Iterable[np.ndarray], dst: np.ndarray, block_size: int):
    if block_size == 1:
        count = -1
        for count, el in enumerate(rows):
            dst[count] = el
        count += 1
    else:
        assert block_size > 0, block_size
        count = 0
        buffer = []
        for row in rows:
            buffer.append(row)
            if len(buffer) == block_size:
                dst[count : count + block_size] = np.stack(buffer, axis=0)
                buffer = []
                count += block_size
        if buffer:
            block_size = len(buffer)
            dst[count : count + block_size] = np.stack(buffer, axis=0)
            count += block_size
    assert count == dst.shape[0], (count, dst.shape[0])

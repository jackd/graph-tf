import abc
import typing as tp

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


class TransitiveData(abc.ABC):
    @property
    @abc.abstractmethod
    def adjacency(self) -> sp.spmatrix:
        pass

    @property
    @abc.abstractmethod
    def node_features(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def labels(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def train_ids(self) -> tp.Optional[np.ndarray]:
        pass

    @property
    @abc.abstractmethod
    def validation_ids(self) -> tp.Optional[np.ndarray]:
        pass

    @property
    @abc.abstractmethod
    def test_ids(self) -> tp.Optional[np.ndarray]:
        pass


class DataSplit(tp.NamedTuple):
    train_data: tf.data.Dataset
    validation_data: tp.Optional[tf.data.Dataset]
    test_data: tp.Optional[tf.data.Dataset]


def data_split(
    train_data: tf.data.Dataset,
    validation_data: tp.Optional[tf.data.Dataset] = None,
    test_data: tp.Optional[tf.data.Dataset] = None,
) -> DataSplit:
    return DataSplit(train_data, validation_data, test_data)

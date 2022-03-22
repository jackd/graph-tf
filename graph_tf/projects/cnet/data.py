import functools
import typing as tp

import gin
import tensorflow as tf

import stfu
from graph_tf.data.single import (
    DataSplit,
    SemiSupervisedSingle,
    preprocess_weights,
    transformed,
)
from graph_tf.utils import graph_utils as gu

register = functools.partial(gin.register, module="gtf.cnet.data")


@register
def preprocess_single(
    data: SemiSupervisedSingle,
    batch_size: int,
    features_transform=(),
    transition_transform=(),
    jl_factor: float = 4.0,
    z_seed: int = 0,
    dataset_seed: int = 0,
    **cg_kwargs
) -> DataSplit:
    Z = gu.approx_effective_resistance_z(
        data.adjacency, jl_factor=jl_factor, rng=z_seed, **cg_kwargs
    )
    # T = gu.get_pairwise_effective_resistance(Z, tf.range(Z.shape[0], dtype=tf.int64))
    # import matplotlib.pyplot as plt
    # import numpy as np

    # # T = tf.where(
    # #     T < 1e-5, tf.fill(tf.shape(T), -tf.constant(np.inf, dtype=T.dtype)), 1 / T
    # # )
    # # T = tf.where(T < 1e-5, tf.zeros_like(T), 1 / T)
    # # T = 1 / (1 + T)
    # # T = tf.math.softmax(T, axis=1)
    # # T = tf.exp(T)
    # T = tf.exp(-T)
    # # T = tf.math.softmax(-T, axis=1)
    # T = T.numpy().reshape(-1)
    # T = np.sort(T)
    # # T = T.numpy()
    # # T = T[T > 1e-4]
    # # T = 1 / T
    # # T = tf.sigmoid(T - 5 * T.mean())
    # # T = np.sort(T)
    # # n = T.shape[0]
    # # T = T[int(0.01 * n) : int(0.99 * n)]
    # print(T.min(), T.max())
    # fig, (ax0, ax1) = plt.subplots(1, 2)
    # ax0.hist(T, bins=50)
    # ax1.hist(T, cumulative=True, bins=50)
    # plt.show()
    # raise Exception("debug")
    # Z = gu.effective_resistance_z(data.adjacency, **cg_kwargs)
    num_nodes = data.adjacency.dense_shape[0]
    features = transformed(data.node_features, features_transform)

    def get_dataset(ids: tp.Optional[tf.Tensor]):
        if ids is None:
            return None
        weights = preprocess_weights(ids, num_nodes, normalize=True)

        def map_fn(ids):
            labels = tf.gather(data.labels, ids, axis=0)
            w = tf.gather(weights, ids, axis=0)
            if isinstance(features, tf.SparseTensor):
                X = stfu.gather(features, ids, axis=0)
            else:
                X = tf.gather(features, ids, axis=0)
            T = gu.get_pairwise_effective_resistance(Z, ids)
            T = tf.math.softmax(-1 * T, axis=1)
            # T = tf.where(T == 0, tf.zeros_like(T), 1 / T)  # resistance to conductance
            # T = transformed(T, transition_transform)
            # T = tf.eye(tf.shape(T)[0])
            inputs = T, X
            return inputs, labels, w

        return (
            tf.data.Dataset.range(num_nodes)
            .shuffle(num_nodes, seed=dataset_seed)
            .batch(batch_size)
            .map(map_fn)
            .prefetch(tf.data.AUTOTUNE)
        )

    return DataSplit(
        get_dataset(data.train_ids),
        get_dataset(data.validation_ids),
        get_dataset(data.test_ids),
    )

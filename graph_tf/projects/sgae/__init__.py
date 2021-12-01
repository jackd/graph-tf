import functools
import typing as tp

import gin
import numpy as np
import tensorflow as tf

from graph_tf.data.single import (
    AutoencoderData,
    AutoencoderDataV2,
    DataSplit,
    EdgeData,
    transformed,
)
from graph_tf.utils.np_utils import random_seed_context
from graph_tf.utils.ops import ravel_multi_index

register = functools.partial(gin.register, module="gtf.sgae")


@register
def sgae(
    inputs_spec: tf.TensorSpec, encoder_fn: tp.Callable[[tf.TensorSpec], tf.keras.Model]
) -> tf.keras.Model:
    encoder = encoder_fn(inputs_spec)
    (inp,) = encoder.inputs
    (z,) = encoder.outputs
    preds = tf.matmul(z, z, transpose_b=True)
    preds = tf.reshape(preds, (-1, 1))
    return tf.keras.Model(inp, preds)


@register
def sgae_v2(
    inputs_spec: tf.TensorSpec, encoder_fn: tp.Callable[[tf.TensorSpec], tf.keras.Model]
) -> tf.keras.Model:
    encoder = encoder_fn(inputs_spec)
    (inp,) = encoder.inputs
    (z,) = encoder.outputs
    z.shape.assert_has_rank(3)
    assert z.shape[1] == 2
    # u, v = tf.unstack(z, axis=1)
    # logits = tf.einsum("bn,bn->b", u, v)
    logits = tf.reduce_sum(tf.reduce_prod(z, axis=1), axis=1)
    logits = tf.expand_dims(logits, -1)
    return tf.keras.Model(inp, logits)


@register
def sgae_v3(
    inputs_spec: tf.TensorSpec, encoder_fn: tp.Callable[[tf.TensorSpec], tf.keras.Model]
) -> tf.keras.Model:
    encoder = encoder_fn(inputs_spec)
    (inp,) = encoder.inputs
    (z,) = encoder.outputs
    z.shape.assert_has_rank(3)
    assert z.shape[1] == 2
    u, v = tf.unstack(z, axis=1)
    logits = tf.matmul(u, v, transpose_b=True)  # [n, n]
    logits = tf.reshape(logits, (-1, 1))  # [n**2, 1]
    return tf.keras.Model(inp, logits)


def eigsh(adj: tf.SparseTensor, *args, seed=0, **kwargs):
    """Deterministic tensorflow wrapper around `scipy.sparse.linalg.eigsh`."""
    # pylint: disable=import-outside-toplevel
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh as eigsh_sp

    # pylint: enable=import-outside-toplevel

    with random_seed_context(seed):
        adj = sp.coo_matrix(
            (adj.values.numpy(), adj.indices.numpy().T), shape=adj.shape
        )
        if "v0" not in kwargs:
            kwargs["v0"] = np.random.normal(size=adj.shape[0]).astype(adj.dtype)
        w, v = eigsh_sp(adj, *args, **kwargs)
        return tf.convert_to_tensor(w, tf.float32), tf.convert_to_tensor(v, tf.float32)


@register
def get_spectral_split(data: AutoencoderData, spectral_size: int) -> DataSplit:
    w, v = eigsh(data.adjacency, spectral_size)
    del w
    if data.features is not None:
        v = tf.concat((v, data.features), 1)

    def get_examples(labels, weights) -> tp.Iterable:
        example = v, labels, weights
        return (example,)

    return DataSplit(
        get_examples(data.train_labels, data.train_weights),
        get_examples(data.true_labels, data.validation_weights),
        get_examples(data.true_labels, data.test_weights),
    )


@register
def get_spectral_split_v2(
    data: AutoencoderDataV2,
    spectral_size: int,
    batch_size: int,
    shuffle_buffer: int = 1024,
    prefetch_buffer: tp.Optional[int] = -1,
) -> DataSplit:
    w, v = eigsh(data.adjacency, spectral_size)
    del w

    def get_examples(
        edge_data: tp.Optional[EdgeData], shuffle_buffer: tp.Optional[int]
    ) -> tp.Iterable:
        if edge_data is None:
            return None
        dataset = tf.data.Dataset.from_tensor_slices(edge_data)
        if shuffle_buffer is not None:
            dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size)

        def map_fn(inputs):
            indices, labels, weights = inputs
            features = tf.gather(v, indices, axis=0)
            return tf.keras.utils.pack_x_y_sample_weight(features, labels, weights)

        dataset = dataset.map(map_fn)
        if prefetch_buffer:
            dataset = dataset.prefetch(prefetch_buffer)
        return dataset

    return DataSplit(
        get_examples(data.train_edges, shuffle_buffer),
        get_examples(data.test_edges, None),  # HACK: reversed validation/test splits
        get_examples(data.validation_edges, None),
    )


@register
def get_spectral_split_v3(
    data: AutoencoderData,
    spectral_size: int,
    batch_size: int,
    shuffle_buffer: int = 256,
    prefetch_buffer: tp.Optional[int] = -1,
    adjacency_transform: tp.Optional[tp.Callable] = None,
) -> DataSplit:
    adjacency = transformed(data.adjacency, adjacency_transform)
    w, v = eigsh(adjacency, spectral_size)
    del w
    num_nodes = v.shape[0]
    rem = num_nodes % batch_size
    padding = batch_size - rem if rem else 0
    num_nodes_ceil = num_nodes + padding
    assert num_nodes_ceil % batch_size == 0, (num_nodes_ceil, batch_size)
    num_parts = num_nodes_ceil // batch_size

    v = tf.pad(v, [[0, 1], [0, 0]])  # pylint: disable=no-value-for-parameter

    def get_examples(labels, weights, shuffle: bool = True):
        if weights is None:
            return None
        # pad so that weights[num_nodes] == 0
        weights = tf.pad(weights, [[0, 1]])  # pylint: disable=no-value-for-parameter
        labels = tf.pad(  # pylint: disable=no-value-for-parameter
            labels, [[0, 1], [0, 0]]
        )

        def generator_fn():
            row_parts = tf.concat(
                (
                    tf.range(num_nodes, dtype=tf.int64),
                    tf.fill((padding,), tf.cast(num_nodes, tf.int64)),
                ),
                0,
            )
            col_parts = row_parts

            if shuffle:
                row_parts = tf.random.shuffle(row_parts)
                col_parts = tf.random.shuffle(col_parts)
            row_parts = tf.reshape(row_parts, (num_parts, batch_size))
            col_parts = tf.reshape(col_parts, (num_parts, batch_size))

            for i in range(num_parts):
                for j in range(num_parts):
                    yield tf.stack((row_parts[i], col_parts[j]), axis=1)

        dataset = tf.data.Dataset.from_generator(
            generator_fn,
            output_signature=tf.TensorSpec((batch_size, 2), dtype=tf.int64),
        )
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_parts ** 2))
        if shuffle and shuffle_buffer:
            dataset = dataset.shuffle(shuffle_buffer)

        def map_fun(indices):
            row_ids, col_ids = tf.unstack(indices, axis=1)
            indices_2d = tf.stack(tf.meshgrid(row_ids, col_ids, indexing="ij"), axis=-1)
            valid = tf.reduce_all(indices_2d < num_nodes, axis=-1)
            indices_1d = ravel_multi_index(
                indices_2d,
                tf.convert_to_tensor((num_nodes, num_nodes), tf.int64),
                axis=-1,
            )
            indices_1d = tf.where(
                valid, indices_1d, num_nodes ** 2 * tf.ones_like(indices_1d)
            )
            indices_1d = tf.reshape(indices_1d, (-1,))  # [batch_size ** 2]
            labels_ = tf.gather(labels, indices_1d, axis=0)
            weights_ = tf.gather(weights, indices_1d, axis=0)
            features = tf.gather(v, indices, axis=0)  # [batch_size, 2, spectral_dim]

            labels_ = tf.expand_dims(labels_, -1)  # [batch_size**2, 1]
            return features, labels_, weights_

        dataset = dataset.map(map_fun)
        if prefetch_buffer:
            dataset = dataset.prefetch(prefetch_buffer)
        return dataset

    return DataSplit(
        get_examples(data.train_labels, data.train_weights, shuffle_buffer),
        get_examples(
            data.true_labels, data.test_weights, shuffle_buffer
        ),  # HACK: reversed validation/test splits
        get_examples(data.true_labels, data.validation_weights, shuffle_buffer),
    )

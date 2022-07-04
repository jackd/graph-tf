import os
import shutil
import typing as tp

import h5py
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags

from graph_tf.projects.igcn.scalable.jl import johnson_lindenstrauss_min_dim
from graph_tf.utils.np_utils import (
    block_column_generator,
    block_row_generator,
    write_block_columns,
)

flags.DEFINE_integer("num_nodes", 130_000, "Total number of nodes")
flags.DEFINE_integer("num_labels", 90_000, "Number of labelled nodes")
flags.DEFINE_integer("num_classes", 40, "Number of labelled nodes")

# flags.DEFINE_integer("num_nodes", 100_000_000, "Total number of nodes")
# flags.DEFINE_integer("num_labels", 100_000, "Number of labelled nodes")
# flags.DEFINE_integer("num_classes", 172, "Number of labelled nodes")


flags.DEFINE_integer("batch_size", 1024, "Batch size")
flags.DEFINE_float("eps", 0.1, "Johnson-Lindenstrauss error")
flags.DEFINE_integer("count", 100, "Number of iterations")
flags.DEFINE_integer("block_size", 1024, "Number of columns to read simultaneously")
flags.DEFINE_integer(
    "write_block_size", 16, "Number of columns to write simultaneously"
)
flags.DEFINE_string("compression", "gzip", "tf compression")


root = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(root, exist_ok=True)


def create_h5_data(
    path: str, num_nodes: int, num_labels: int, num_classes: int, eps: float
):
    if os.path.exists(path):
        os.remove(path)
    k = johnson_lindenstrauss_min_dim(num_labels, eps=eps)
    with h5py.File(path, "a") as root:
        data = root.create_dataset("ra", shape=(k, num_nodes), dtype=np.float32)
        for i in tqdm.trange(k, desc="Creating ra data"):
            data[i] = np.random.normal(size=(num_nodes,)).astype(np.float32)
        data = root.create_dataset("ay", shape=(num_classes, num_nodes))
        for i in tqdm.trange(num_classes, desc="Creating ay data"):
            data[i] = np.random.normal(size=(num_nodes,)).astype(np.float32)


def create_h5_transpose_data(
    src_path: str,
    dst_path: str,
    block_size: int,
):
    if os.path.exists(dst_path):
        os.remove(dst_path)
    with h5py.File(dst_path, "a") as dst:
        with h5py.File(src_path, "r") as src:
            for k, v in src.items():
                data = dst.create_dataset(k, shape=v.shape[-1::-1], dtype=v.dtype)
                for i, vi in enumerate(
                    tqdm.tqdm(
                        block_column_generator(v, block_size=block_size),
                        desc=f"Writing {k} ",
                        total=v.shape[1],
                    )
                ):
                    data[i] = vi


def iterate_lazy(count: int, path: str, block_size: int):
    with h5py.File(path, "r") as root:
        ra = root["ra"]
        ay = root["ay"]
        k = ra.shape[0]  # pylint: disable=no-member
        num_classes = ay.shape[0]  # pylint: disable=no-member

        def gen():
            for rai, ayi in zip(
                block_column_generator(ra, block_size),
                block_column_generator(ay, block_size),
            ):
                yield rai, ayi

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec((k,), dtype=tf.float32),
                tf.TensorSpec((num_classes,), dtype=tf.float32),
            ),
        )

        for _ in tqdm.tqdm(
            dataset.take(count), total=count, desc="Iterating generator dataset"
        ):
            pass


def iterate_lazy_transpose(count: int, path: str, block_size: int):
    with h5py.File(path, "r") as root:
        ra = root["ra"]
        ay = root["ay"]
        k = ra.shape[1]  # pylint: disable=no-member
        num_classes = ay.shape[1]  # pylint: disable=no-member

        def gen():
            for rai, ayi in zip(
                block_row_generator(ra, block_size),
                block_row_generator(ay, block_size),
            ):
                yield rai, ayi

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec((k,), dtype=tf.float32),
                tf.TensorSpec((num_classes,), dtype=tf.float32),
            ),
        )

        for _ in tqdm.tqdm(
            dataset.take(count),
            total=count,
            desc="Iterating transpose generator dataset",
        ):
            pass


def iterate_lazy_batched(count: int, path: str, batch_size: int):
    with h5py.File(path, "r") as root:
        ra = root["ra"]
        ay = root["ay"]
        k, num_nodes = ra.shape  # pylint: disable=no-member
        num_classes = ay.shape[0]  # pylint: disable=no-member

        def np_fn(indices):
            return ra[:, indices], ay[:, indices]

        def map_fn(seed):
            _, indices = tf.nn.top_k(
                tf.random.stateless_uniform((num_nodes,), seed), batch_size
            )
            indices = tf.sort(indices)
            ra, ay = tf.numpy_function(
                np_fn, (indices,), (tf.float32, tf.float32), stateful=False
            )
            ra.set_shape((k, batch_size))
            ay.set_shape((num_classes, batch_size))
            return tf.transpose(ra), tf.transpose(ay)

        dataset = tf.data.Dataset.random(0).batch(2).take(count).map(map_fn).prefetch(1)
        for _ in tqdm.tqdm(
            dataset,
            total=count,
            desc=f"Iterating generator dataset (batch_size={batch_size})",
        ):
            pass


def iterate_lazy_transpose_batched(count: int, path: str, batch_size: int):
    with h5py.File(path, "r") as root:
        ra = root["ra"]
        ay = root["ay"]
        num_nodes, k = ra.shape  # pylint: disable=no-member
        num_classes = ay.shape[1]  # pylint: disable=no-member

        def np_fn(indices):
            return ra[indices], ay[indices]

        def map_fn(seed):
            _, indices = tf.nn.top_k(
                tf.random.stateless_uniform((num_nodes,), seed), batch_size
            )
            indices = tf.sort(indices)
            ra, ay = tf.numpy_function(
                np_fn, (indices,), (tf.float32, tf.float32), stateful=False
            )
            ra.set_shape((batch_size, k))
            ay.set_shape((batch_size, num_classes))
            return ra, ay

        dataset = tf.data.Dataset.random(0).batch(2).take(count).map(map_fn).prefetch(1)
        for _ in tqdm.tqdm(
            dataset,
            total=count,
            desc=f"Iterating transpose generator dataset (batch_size={batch_size})",
        ):
            pass


# def iterate_tfio_transpose_batched(count: int, path: str, batch_size: int):
#     data = tfio.IOTensor.from_hdf5(path)
#     ra = data(b'/ra')
#     ay = data(b'/ay')

#     def map_fn(seed):


def create_saved_dataset(
    h5_path: str, save_path: str, block_size: int, compression: tp.Optional[str] = None
):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    with h5py.File(h5_path, "r") as root:
        ra = root["ra"]
        ay = root["ay"]
        k, num_nodes = ra.shape  # pylint: disable=no-member
        num_labels = ay.shape[0]  # pylint: disable=no-member

        def gen():
            for rai, ayi in tqdm.tqdm(
                zip(
                    block_column_generator(ra, block_size),
                    block_column_generator(ay, block_size),
                ),
                total=num_nodes,
                desc="Creating tf_dataset",
            ):
                yield rai, ayi

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec((k,), dtype=tf.float32),
                tf.TensorSpec((num_labels,), dtype=tf.float32),
            ),
        )
        tf.data.experimental.save(dataset, save_path, compression=compression)


def iterate_saved(
    count: int, save_path: str, compression: tp.Optional[str] = None, batch_size=None
):
    dataset = tf.data.experimental.load(save_path, compression=compression)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
        desc = f"Iterating saved dataset (batch_size = {batch_size}"
    else:
        desc = "Iterating saved dataset (unbatched)"
    for _ in tqdm.tqdm(dataset.take(count), desc=desc):
        pass


def main(_):
    FLAGS = flags.FLAGS
    h5_path = os.path.join(root, "base.h5")
    h5t_path = os.path.join(root, "transpose.h5")
    tf_path = os.path.join(root, "saved")
    compression = FLAGS.compression.upper()

    # create data
    create_h5_data(
        h5_path, FLAGS.num_nodes, FLAGS.num_labels, FLAGS.num_classes, FLAGS.eps
    )
    create_h5_transpose_data(
        h5_path,
        h5t_path,
        FLAGS.block_size,
    )
    create_saved_dataset(
        h5_path, tf_path, block_size=FLAGS.block_size, compression=compression
    )

    count = FLAGS.count * FLAGS.batch_size
    iterate_lazy(count, h5_path, block_size=FLAGS.block_size)
    iterate_lazy_transpose(count, h5t_path, block_size=FLAGS.block_size)
    iterate_saved(count, tf_path, compression=compression)

    # batched
    iterate_lazy_batched(FLAGS.count, h5_path, FLAGS.batch_size)
    iterate_lazy_transpose_batched(FLAGS.count, h5t_path, FLAGS.batch_size)
    iterate_saved(
        FLAGS.count, tf_path, batch_size=FLAGS.batch_size, compression=compression
    )


if __name__ == "__main__":
    app.run(main)

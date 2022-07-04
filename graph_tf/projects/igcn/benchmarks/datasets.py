"""
With default clargs:

```txt
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy
and will incur extra overhead.
Writing h5py dataset: 100%|████████████| 100000/100000 [00:08<00:00, 12131.80it/s]
Writing h5py dataset: 100%|████████████| 100000/100000 [00:08<00:00, 12020.78it/s]
Writing in-memory dataset: 100%|███████| 100000/100000 [00:01<00:00, 84822.50it/s]
Writing in-memory dataset: 100%|███████| 100000/100000 [00:01<00:00, 87337.17it/s]
Writing in-memory dataset: 100%|███████| 100000/100000 [00:01<00:00, 85142.96it/s]
Writing memmap dataset: 100%|██████████| 100000/100000 [00:01<00:00, 51721.00it/s]
Writing memmap dataset: 100%|██████████| 100000/100000 [00:01<00:00, 51362.74it/s]
Writing memmap dataset: 100%|██████████| 100000/100000 [00:01<00:00, 51381.38it/s]
-----------------------------------------------------
Benchmark           Time             CPU   Iterations
-----------------------------------------------------
hdf5       2119862294 ns      3524585 ns           10
in_memory    30939472 ns      1123062 ns          100
memmap       30487819 ns      1060036 ns          100
```

i.e. `hdf5` is ~70x slower than `memmap`, which isn't similar in speed to `in_memory`.
"""
import os

import google_benchmark as benchmark
import numpy as np
import tensorflow as tf
from absl import flags

from graph_tf.utils import io_utils
from graph_tf.utils.temp_utils import tempfile_context

flags.DEFINE_integer("samples", 10_000, "number of samples")
flags.DEFINE_integer("rows", 100_000, "number of rows")
# flags.DEFINE_integer("samples", 500, "number of samples")
# flags.DEFINE_integer("rows", 1_000, "number of rows")
flags.DEFINE_integer("cols", 1_000, "number of columns")
flags.DEFINE_integer("seed", 0, "rng seed")
flags.DEFINE_integer("num_parallel_calls", 1, "used in `Dataset.map`")
flags.DEFINE_integer("prefetch_buffer", 1, "used in `Dataset.prefetch`")

FLAGS = flags.FLAGS


def print_size(path: str, desc: str):
    st = os.stat(path)
    print(f"{desc} size (B): {st.st_blocks * 512}")


def get_base_data() -> tf.data.Dataset:
    def map_fn(seed):
        return tf.random.stateless_uniform(
            shape=(FLAGS.cols,), dtype=tf.float32, seed=seed
        )

    return tf.data.Dataset.random(FLAGS.seed).batch(2).take(FLAGS.rows).map(map_fn)


def get_dataset(data: np.ndarray) -> tf.data.Dataset:
    def map_fn(seed):
        _, indices = tf.nn.top_k(
            tf.random.stateless_uniform((FLAGS.rows,), seed=seed), FLAGS.samples
        )
        indices = tf.sort(indices)
        out: tf.Tensor = tf.numpy_function(
            lambda indices: data[indices], (indices,), tf.float32, stateful=False
        )
        out.set_shape((FLAGS.samples, FLAGS.cols))
        return out

    return (
        tf.data.Dataset.random(FLAGS.seed)
        .batch(2)
        .map(map_fn, FLAGS.num_parallel_calls)
        .prefetch(FLAGS.prefetch_buffer)
    )


def run_benchmark(state, writer: io_utils.Writer, reader: io_utils.Reader):

    with tempfile_context() as path:
        path = os.path.join(path, "data")
        os.makedirs(path)
        writer(path, get_base_data())
        dataset = get_dataset(reader(path))

        @tf.function
        def f(it):
            el = next(it)
            return tf.reduce_sum(el)

        it = iter(dataset)
        f(it)

        while state:
            f(it).numpy()


@benchmark.register()
def hdf5(state):
    run_benchmark(state, io_utils.write_hdf5, io_utils.read_hdf5)


@benchmark.register()
def in_memory(state):
    run_benchmark(state, io_utils.write_numpy, io_utils.read_numpy)


@benchmark.register()
def memmap(state):
    run_benchmark(state, io_utils.write_memmap, io_utils.read_memmap)


# @benchmark.register()
# def zarr(state):
#     run_benchmark(state, io_utils.write_zarr, io_utils.read_zarr)


if __name__ == "__main__":
    benchmark.main()

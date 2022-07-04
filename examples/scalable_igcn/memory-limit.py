"""
Example usage

```bash
python memory-limit.py --tb=/tmp/igcn-tb --batch_size=1024
tensorboard --logdir=/tmp/igcn-tb
```

GPU Memory usage according to tensorboard `memory_profile`:
   bs: memory usage  step time (on GTX-1050Ti)
 1024:          59M       14.7ms
 2048:         110M       25.5ms
 4096:         214M       47.6ms
 8192:         425M       92.4ms
16384:         839G      183.8ms
32768:            G            s


V2
```bash
python memory-limit.py --tb=/tmp/igcn-tb --v2 --batch_size=1024
```
GPU Memory usage according to tensorboard `memory_profile`:
   bs: memory usage  step time (on GTX-1050Ti)
 1024:          32M        6.8ms
 2048:          77M       15.7ms
 4096:         251M       45.1ms
 8192:       1.110G      157.9ms
Quadratic projections
16384:       4    G      640  ms
32768:      16    G    2.5     s

max on P100 (10GB): 16384
max on GTX-1050Ti (4GB): 8192  (maybe cpu memory bound?)
"""
import os

import tensorflow as tf
from absl import app, flags

from graph_tf.projects.igcn.scalable.jl import johnson_lindenstrauss_min_dim
from graph_tf.projects.igcn.scalable.losses import (
    LazyQuadraticCategoricalCrossentropy,
    LazyQuadraticCrossentropyLabelData,
)
from graph_tf.utils import models as model_utils

flags.DEFINE_integer("batch_size", default=1024, help="batch size")
flags.DEFINE_integer("num_labels", default=100_000, help="Number of labels")
flags.DEFINE_integer("num_features", default=128, help="Number of features")
flags.DEFINE_integer("hidden_units", default=256, help="Number of hidden units")
flags.DEFINE_integer("num_layers", default=3, help="Number of hidden layers")
flags.DEFINE_integer("num_classes", default=40, help="Number of classes")
flags.DEFINE_float("eps", default=0.1, help="Johnson-Lindenstrauss error")
flags.DEFINE_string("tb", default="", help="tensorboard directory")
flags.DEFINE_boolean("v2", default=False, help="Whether or not to use v2")


def main(_):
    FLAGS = flags.FLAGS
    batch_size = FLAGS.batch_size
    num_labels = FLAGS.num_labels
    eps = FLAGS.eps
    hidden_units = (FLAGS.hidden_units,) * FLAGS.num_layers
    num_features = FLAGS.num_features
    num_classes = FLAGS.num_classes
    tb_dir = os.path.expanduser(os.path.expandvars(FLAGS.tb))
    tb_dir = os.path.join(tb_dir, "v2" if FLAGS.v2 else "v1")

    dropout_rate = 0.5
    k = johnson_lindenstrauss_min_dim(num_labels, eps=eps)

    mlp = model_utils.mlp(
        tf.TensorSpec((None, num_features), dtype=tf.float32),
        output_units=num_classes,
        hidden_units=hidden_units,
        normalization=model_utils.batch_norm,
        dropout_rate=dropout_rate,
    )
    mlp.compile(
        loss=LazyQuadraticCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
    )

    def map_fn(seed):
        seeds = tf.unstack(tf.random.experimental.stateless_split(seed, num=3), axis=0)

        features = tf.random.stateless_normal(
            (batch_size, num_features), seed=seeds.pop()
        )
        data = LazyQuadraticCrossentropyLabelData(
            tf.random.stateless_uniform((batch_size, num_classes), seed=seeds.pop()),
            tf.random.stateless_uniform((batch_size, k), seed=seeds.pop()),
        )
        if FLAGS.v2:
            data = data.to_v2()
        assert not seeds
        return features, data

    dataset = tf.data.Dataset.random(0).batch(2).map(map_fn).prefetch(1)

    callbacks = []
    if tb_dir:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                os.path.join(tb_dir, f"bs-{FLAGS.batch_size}"), profile_batch=(2, 5)
            )
        )
    mlp.fit(dataset, callbacks=callbacks, epochs=2, steps_per_epoch=5)


if __name__ == "__main__":
    app.run(main)

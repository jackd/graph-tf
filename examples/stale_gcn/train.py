import functools
import os

import h5py
import numpy as np
import tensorflow as tf

from graph_tf.data.single import SemiSupervisedSingle, citations_data
from graph_tf.data.transforms import (
    add_identity,
    normalize_symmetric,
    row_normalize,
    to_format,
)
from graph_tf.projects.stale_gcn.models import gcn
from graph_tf.projects.stale_gcn.train import StaleTrainer

cache_dir = "/tmp/stale_gcn_cora"
batch_size = 1
# batch_size = 2708
use_dense_adjacency = False
# use_dense_adjacency = True
epochs = 2000
seed = 0
l2_reg = 2.5e-4
lr = 1e-2
patience = 10
num_runs = 10

tf.random.set_seed(seed)

data = citations_data("cora")
num_nodes = data.adjacency.shape[0]
batch_frac = batch_size / num_nodes
steps_per_epoch = num_nodes // batch_size + int(num_nodes % batch_size > 0)

# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#     [10 * steps_per_epoch], [1e-3, 1e-2]
# )

os.makedirs(cache_dir, exist_ok=True)


data = SemiSupervisedSingle(
    row_normalize(to_format(data.node_features, "dense")),
    normalize_symmetric(add_identity(data.adjacency)),
    data.labels,
    data.train_ids,
    data.validation_ids,
    data.test_ids,
)


def get_compile_kwargs():
    return dict(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="sum"
        ),
        metrics=(),
        weighted_metrics=(
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseCategoricalCrossentropy(
                name="cross_entropy", from_logits=True
            ),
        ),
    )


base_model_fn = functools.partial(
    gcn, num_classes=7, hidden_filters=(16,), dropout_rate=0.5
)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        restore_best_weights=True, monitor="val_cross_entropy", patience=patience
    )
]


def do_stale_run(seed, cache_path):
    print("Starting stale run")
    print(f"seed = {seed}")
    print(f"cache_path = {cache_path}")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    assert not os.path.exists(cache_path)
    cache = h5py.File(cache_path, "a")

    tf.random.set_seed(seed)
    trainer = StaleTrainer(
        data,
        functools.partial(base_model_fn, l2_reg=l2_reg * batch_frac),
        cache,
        train_batch_size=batch_size,
        cache_batch_size=-1,  # full batch
        use_dense_adjacency=use_dense_adjacency,
        **get_compile_kwargs(),
    )

    trainer.fit(epochs, callbacks=callbacks)
    result = trainer.test()
    print(result)
    os.remove(cache_path)
    return result


accs = [
    do_stale_run(s, os.path.join(cache_dir, f"cache-{s}.h5"))["acc"]
    for s in range(num_runs)
]
print(np.mean(accs), np.std(accs))

## fresh model
# tf.random.set_seed(seed)
# build_and_fit(
#     to_classification_split(data),
#     functools.partial(base_model_fn, l2_reg=l2_reg),
#     callbacks=callbacks,
#     epochs=epochs,
#     **get_compile_kwargs(),
# )

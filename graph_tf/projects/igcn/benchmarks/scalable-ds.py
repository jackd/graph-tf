import os
import tempfile

import tqdm

from graph_tf.projects.igcn import scalable_backup

# batch_size = 1024 * 16
label_batch_size = 1024
feature_batch_size = 1024
path = os.path.join(
    os.path.expanduser(os.path.expandvars("$GTF_DATA_DIR")),
    "igcn",
    "scalable",
    "ogbn-arxiv.h5",
)
# cache = get_cache(path, mode="a")
# for split in ("train", "validation"):
#     cache.prepare_transition(split, epsilon=0.1)
# cache.close()
# exit()

cache = scalable_backup.get_cache(path, mode="r")
print("Loading train dataset...")
# train_dataset, _ = load_train_dataset(cache, batch_size, 0.1, temperature=0.1)
train_dataset, steps_per_epoch = scalable_backup.load_train_dataset_v1(
    cache=cache,
    label_batch_size=label_batch_size,
    feature_batch_size=feature_batch_size,
    epsilon=0.1,
    temperature=0.1,
)
train_dataset = train_dataset.take(steps_per_epoch)
print("  Finished")
print("Iterating...")
# for el in tqdm.tqdm(train_dataset.prefetch(-1), desc="base"):
#     pass

# compression = "cache"
compression = "GZIP"
path = os.path.join(tempfile.tempdir, "tf-snapshots", str(compression))
# for el in tqdm.tqdm(train_dataset.cache(path).prefetch(-1), desc="Cached, iter 0"):
#     pass
if compression == "cache":
    snapshot = train_dataset.cache(path)
else:
    snapshot = train_dataset.snapshot(path, compression=compression)
for el in tqdm.tqdm(
    snapshot.prefetch(-1),
    desc=f"Snapshot {compression}, iter 0",
):
    pass

for el in tqdm.tqdm(
    snapshot.prefetch(-1),
    desc=f"Snapshot {compression}, iter 1",
):
    pass

print(path)

# with scalable.tempfile_context() as path:
#     train_dataset = train_dataset.cache(path)

#     for el in tqdm.tqdm(train_dataset.prefetch(-1), desc="Cached, iter 0"):
#         pass

#     for el in tqdm.tqdm(train_dataset.prefetch(-1), desc="Cached, iter 1"):
#         pass

# def map_fn(inputs, labels, weights=None):
#     T, x = inputs
#     T = tf.matmul(T, T, transpose_b=True)
#     # T = tf.numpy_function(lambda x: x @ x.T, (T,), tf.float32, stateful=False)
#     T.set_shape((batch_size, batch_size))
#     return tf.keras.utils.pack_x_y_sample_weight((T, x), labels, weights)


# train_dataset = train_dataset.map(map_fn)

# for el in tqdm.tqdm(train_dataset.prefetch(-1).take(iters), desc="Mapped"):
#     pass

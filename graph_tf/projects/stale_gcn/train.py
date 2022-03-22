import functools
import itertools
import os
import typing as tp
from collections import deque

import gin
import h5py
import keras
import numpy as np
import tensorflow as tf

from graph_tf.data.single import SemiSupervisedSingle, preprocess_weights
from graph_tf.mains.build_and_fit import repeat
from graph_tf.projects.stale_gcn.layers import Propagation, StalePropagation
from stfu.ops import to_dense_index_lookup

# from stfu.ops import dense_hash_table_index_lookup


KerasTensor = keras.backend.keras_tensor.KerasTensor
Node = keras.engine.node.Node

T = tp.TypeVar("T")

configurable = functools.partial(gin.configurable, module="gtf.stale_gcn.train")
register = functools.partial(gin.register, module="gtf.stale_gcn.train")


def dfs(
    start: tp.Iterable[T],
    neighbors_fn: tp.Callable[[T], tp.Iterable[T]],
    key_fn: lambda x: x,
):
    stack = list(start)
    visited = set()
    while stack:
        n = stack.pop()
        key = key_fn(n)
        if key not in visited:
            visited.add(key)
            yield n
            stack.extend(neighbors_fn(n))


def bfs(
    start: tp.Iterable[T],
    neighbors_fn: tp.Callable[[T], tp.Iterable[T]],
    key_fn: lambda x: x,
):
    q = deque(start)
    visited = set()
    while q:
        n = q.popleft()
        key = key_fn(n)
        if key not in visited:
            visited.add(key)
            yield n
            q.extend(neighbors_fn(n))


def _tensor_dependencies(tensor: KerasTensor, visited: tp.Set):
    ref = tensor.ref()
    if ref not in visited:
        visited.add(ref)
        if isinstance(tensor, KerasTensor):  # reject `EagerTensor`s
            for input_tensor in tf.nest.flatten(tensor.node.input_tensors):
                yield from _tensor_dependencies(input_tensor, visited)
            yield tensor


def tensor_dependencies(tensors: tp.Iterable[KerasTensor]):
    visited = set()
    for t in tensors:
        yield from _tensor_dependencies(t, visited)


# def keras_history(tensor: KerasTensor) -> KerasHistory:
#     return tensor._keras_history  # pylint: disable=protected-access


# def inbound_nodes(history: KerasHistory) -> tp.Sequence[Node]:
#     return history._inbound_nodes  # pylint: disable=protected-access


# def layer_neighbors(layer: tf.keras.layers.Layer) -> tp.Iterable[tf.keras.Layer]:
#     for node in layer.inbound_nodes:
#         yield from (keras_history(i).layer for i in node.input_nodes)


def tensor_neighbors(tensor: KerasTensor) -> tp.Iterable[KerasTensor]:
    return tf.nest.flatten(tensor.node.input_tensors)


# @register
# def build(inputs_spec, output_fn: tp.Callable) -> tf.keras.Model:
#     inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), inputs_spec)
#     outputs = output_fn(*inputs)

#     propagations = tuple(
#         l
#         for l in bfs(
#             (keras_history(o).layer for o in tf.nest.flatten(outputs)),
#             layer_neighbors
#         )
#         if isinstance(l, Propagation)
#     )[-1::-1]
#     prop_inputs = {p.name: {"x0": p.x0, "y0": p.y0} for p in propagations}
#     return tf.keras.Model((inputs, prop_inputs), outputs)


def gather_gather(st: tf.SparseTensor, indices: tf.Tensor):
    # out_index_map = dense_hash_table_index_lookup(
    #     indices, tf.range(st.dense_shape[0], dtype=st.indices.dtype)
    # )
    out_index_map = to_dense_index_lookup(
        indices, tf.range(st.dense_shape[0], dtype=st.indices.dtype), st.dense_shape[0]
    )
    out_indices = tf.gather(out_index_map, st.indices, axis=0)
    valid = tf.reduce_all(out_indices >= 0, axis=-1)
    out_values = tf.boolean_mask(st.values, valid)
    out_indices = tf.boolean_mask(out_indices, valid)
    out_shape = tf.tile(tf.shape(indices, out_type=tf.int64), [2])
    st = tf.SparseTensor(out_indices, out_values, out_shape)
    return st


def build_stale_dataset(
    stale_model: tf.keras.Model,
    data: SemiSupervisedSingle,
    cache: h5py.Group,
    batch_size: int,
    *,
    use_dense_adjacency: bool = False,
) -> tf.data.Dataset:
    def get_key(name):
        assert name.startswith("stale_")
        return name[len("stale_") :]

    keys = tuple(
        get_key(layer.name)
        for layer in stale_model.layers
        if isinstance(layer, StalePropagation)
    )

    def load_prop(key, ids):
        group = cache[key]
        # TODO: make these slices rather than gathers
        return np.array(group["x0"][ids]), np.array(group["y0"][ids])

    def load_stale_values(ids):
        return tuple(itertools.chain(*(load_prop(key, ids) for key in keys)))

    def map_fn(ids):
        # TODO: parallelize loading?
        ids = tf.sort(ids)
        dtypes = itertools.chain(
            *((cache[k]["x0"].dtype, cache[k]["y0"].dtype) for k in keys)
        )
        stale_values = tf.numpy_function(
            load_stale_values, [ids], [tf.dtypes.as_dtype(d) for d in dtypes],
        )
        x0s = stale_values[::2]
        y0s = stale_values[1::2]
        stale_values = {}

        for key, x0, y0 in zip(keys, x0s, y0s):
            x0.set_shape((None, cache[key]["x0"].shape[1]))
            y0.set_shape((None, cache[key]["y0"].shape[1]))
            stale_values[key] = {"x0": x0, "y0": y0}

        features = tf.gather(data.node_features, ids, axis=0)
        adj = gather_gather(data.adjacency, ids)
        if use_dense_adjacency:
            adj = tf.sparse.to_dense(adj)
        inputs = (features, adj), stale_values
        labels = tf.gather(data.labels, ids, axis=0)
        batch_weights = tf.gather(weights, ids, axis=0)
        return inputs, labels, batch_weights

    num_nodes = data.adjacency.shape[0]
    weights = preprocess_weights(data.train_ids, num_nodes, normalize=True)
    ids_ds = tf.data.Dataset.range(num_nodes).shuffle(num_nodes).batch(batch_size)
    dataset = ids_ds.map(map_fn)
    tf.nest.assert_same_structure(dataset.element_spec[0], stale_model.input)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_stale_model(model: tf.keras.Model, *, use_dense_adjacency: bool = False):
    features, adj = model.input
    stale_features = tf.keras.Input(
        type_spec=features.type_spec, name=f"{features.node.layer.name}_stale",
    )
    if use_dense_adjacency:
        kwargs = dict(shape=adj.shape[1:], batch_size=adj.shape[0], dtype=adj.dtype)
    else:
        kwargs = dict(type_spec=adj.type_spec)
    stale_adj = tf.keras.Input(name=f"{adj.node.layer.name}_stale", **kwargs)
    stale_layer_inputs = {}
    orig_to_stale = {features.ref(): stale_features, adj.ref(): stale_adj}

    def get_stale(orig):
        if isinstance(orig, KerasTensor):
            return orig_to_stale[orig.ref()]
        return orig

    # tensors = tuple(
    #     bfs(tf.nest.flatten(model.output), tensor_neighbors, key_fn=lambda x: x.ref())
    # )[-1::-1]
    tensors = tuple(tensor_dependencies(tf.nest.flatten(model.output)))

    for tensor in tensors:
        if tensor.ref() in orig_to_stale:
            continue
        node = tensor.node
        layer = node.layer
        inputs = tf.nest.map_structure(get_stale, node.input_tensors)
        if isinstance(layer, Propagation):
            stale_layer = StalePropagation(layer, name=f"stale_{layer.name}")
            outputs = stale_layer.build_and_call(*inputs)
            stale_layer_inputs[layer.name] = {
                "x0": stale_layer.x0,
                "y0": stale_layer.y0,
            }
        else:
            outputs = layer(inputs)
        flat_node_outputs = tf.nest.flatten(node.output_tensors)
        flat_outputs = tf.nest.flatten(outputs)
        assert len(flat_node_outputs) == len(flat_outputs)
        for orig, stale in zip(flat_node_outputs, flat_outputs):
            orig_to_stale[orig.ref()] = stale

    stale_inputs = tf.nest.map_structure(get_stale, model.input)
    stale_inputs = stale_inputs, stale_layer_inputs
    outputs = tf.nest.map_structure(get_stale, model.output)
    stale_model = tf.keras.Model(stale_inputs, outputs,)
    return stale_model


def _get_input_model(outputs: KerasTensor, base_input) -> tf.keras.Model:
    """
    Get a keras model that maps base node features and propagator outputs to x.

    Args:
        x: a Propagator input.

    Returns:
        model, such that model(mapping) = x, and mapping is a dict from string keys
            'base' or p.name to base node features or propagator outputs respectively.

    """

    def is_prop_output(x):
        return isinstance(x.node.layer, Propagation)

    tensors = tuple(
        bfs(
            tf.nest.flatten(outputs),
            lambda x: () if is_prop_output(x) else tensor_neighbors(x),
            key_fn=lambda x: x.ref(),
        )
    )
    inputs = {t.node.layer.name: t for t in tensors[-1::-1] if is_prop_output(t)}
    if base_input.name in [t.name for t in tensors]:
        inputs["base"] = base_input
    # use identity in case outputs are also inputs
    outputs = tf.nest.map_structure(tf.identity, outputs)
    model = tf.keras.Model(inputs, outputs)
    assert not any(isinstance(layer, Propagation) for layer in model.layers)
    return model


def _get_input_dataset(
    model: tf.keras.Model,
    data: SemiSupervisedSingle,
    cache: h5py.Group,
    batch_size: int,
) -> tf.data.Dataset:
    num_nodes = data.adjacency.shape[0]
    if batch_size == -1:
        batch_size = num_nodes
    keys = [k for k in model.input if k != "base"]

    def update_shapes(outputs):
        for o, k in zip(outputs, keys):
            o.set_shape((None, cache[k]["y0"].shape[1]))

    if batch_size == -1:

        def generator():
            mapping = {k: tf.convert_to_tensor(cache[k]["y0"]) for k in keys}
            if "base" in model.input:
                mapping["base"] = data.node_features
            yield mapping

        mapping_spec = {
            k: tf.TensorSpec(cache[k]["y0"].shape, cache[k]["y0"].dtype) for k in keys
        }
        if "base" in model.input:
            mapping_spec["base"] = tf.TensorSpec(
                shape=data.node_features.shape, dtype=data.node_features.dtype
            )
        dataset = tf.data.Dataset.from_generator(
            generator, output_signature=mapping_spec
        )
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(1))
    else:

        def load_cache_outputs(ids):
            # TODO: make this a slice rather than a gather
            return tuple(cache[k]["y0"][ids] for k in keys)

        def map_fn(ids):
            # TODO: parallelize loading?
            outputs = tf.numpy_function(
                load_cache_outputs, (ids,), [cache[k]["y0"].dtype for k in keys],
            )
            outputs = tf.nest.flatten(outputs)
            update_shapes(outputs)
            mapping = {k: v for k, v in zip(keys, outputs)}
            if "base" in model.input:
                mapping["base"] = tf.gather(data.node_features, ids, axis=0)
            return mapping, ids

        ids_ds = tf.data.Dataset.range(num_nodes).batch(batch_size)
        dataset = ids_ds.map(map_fn)

    tf.nest.assert_same_structure(dataset.element_spec[0], model.input)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


class InMemoryGroup:
    def __init__(self):
        self._data = {}

    def require_group(self, key):
        if key not in self._data:
            self._data[key] = InMemoryGroup()
        else:
            assert isinstance(self._data[key], InMemoryGroup)
        return self._data[key]

    def require_dataset(self, key, shape, dtype):
        if key not in self._data:
            self._data[key] = np.zeros(shape=shape, dtype=dtype)
        else:
            assert isinstance(self._data[key], np.ndarray)
        return self._data[key]

    def __getitem__(self, key):
        return self._data[key]


@register
class StaleTrainer:
    def __init__(
        self,
        data: SemiSupervisedSingle,
        model_fn: tp.Callable[[tp.Any], tf.keras.Model],
        train_batch_size: int,
        cache_batch_size: int,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
        metrics: tp.Sequence[tf.keras.metrics.Metric] = (),
        weighted_metrics: tp.Sequence[tf.keras.metrics.Metric] = (),
        cache: tp.Optional[tp.Union[h5py.Group, InMemoryGroup]] = None,
        use_dense_adjacency: bool = False,
    ):
        if cache is None:
            cache = InMemoryGroup()
        num_nodes = data.adjacency.shape[0]
        self.data = data
        if train_batch_size == -1:
            train_batch_size = num_nodes

        features_spec = tf.TensorSpec(
            shape=(None, *data.node_features.shape[1:]), dtype=data.node_features.dtype
        )
        adj_spec = (
            tf.SparseTensorSpec
            if isinstance(data.adjacency, tf.SparseTensor)
            else tf.TensorSpec
        )(shape=(None, None), dtype=data.adjacency.dtype)
        spec = (features_spec, adj_spec)
        self.model = model_fn(spec)
        features, adj = self.model.input
        self.cache = cache
        self._propagations = tuple(
            p for p in self.model.layers if isinstance(p, Propagation)
        )
        for p in self._propagations:
            g = cache.require_group(p.name)
            for key in ("x0", "y0"):
                g.require_dataset(
                    key, shape=(num_nodes, p.input[1].shape[-1]), dtype=p.dtype,
                )
                g[key][:] = 0

        self.stale_model = build_stale_model(
            self.model, use_dense_adjacency=use_dense_adjacency
        )
        self.stale_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
        )
        self.stale_dataset = build_stale_dataset(
            self.stale_model,
            data,
            cache,
            train_batch_size,
            use_dense_adjacency=use_dense_adjacency,
        )

        self._input_models = {}
        self._input_datasets = {}
        for p in self._propagations:
            model = _get_input_model(p.input[1], features)
            assert adj.ref() not in [t.ref() for t in tf.nest.flatten(model.input)]
            self._input_models[p.name] = model
            self._input_datasets[p.name] = _get_input_dataset(
                model, data, cache, cache_batch_size
            )

        self._output_model = _get_input_model(self.model.output, features)
        self._output_model.compile(
            loss=loss, metrics=metrics, weighted_metrics=weighted_metrics
        )
        output_dataset = _get_input_dataset(
            self._output_model, data, cache, cache_batch_size
        )

        def val_dataset(split_ids):
            weights = preprocess_weights(split_ids, data.adjacency.shape[0])

            def map_fn(mapping, ids):
                batch_labels = tf.gather(data.labels, ids, axis=0)
                batch_weights = tf.gather(weights, ids, axis=0)
                return mapping, batch_labels, batch_weights

            return output_dataset.map(map_fn)

        self._validation_ds = val_dataset(data.validation_ids)
        self._test_ds = val_dataset(data.test_ids)

    def _update_outputs(self, prop: Propagation):
        # TODO: split this up or use cpu.
        g = self.cache[prop.name]
        g["y0"][:] = prop((self.data.adjacency, tf.convert_to_tensor(g["x0"]))).numpy()

    def _update_inputs(self, prop: Propagation):
        name = prop.name
        model = self._input_models[name]
        dataset = self._input_datasets[name]
        group = self.cache[name]["x0"]
        if hasattr(dataset.element_spec, "items"):
            for mapping in dataset:
                group[:] = model(mapping).numpy()
        else:
            for mapping, ids in dataset:
                group[ids.numpy()] = model(mapping).numpy()

    def update_cache(self):
        for p in self._propagations:
            self._update_inputs(p)
            self._update_outputs(p)

    def fit(
        self,
        epochs: int = 1,
        verbose: bool = True,
        callbacks: tp.Sequence[tf.keras.callbacks.Callback] = (),
        initial_epoch: int = 0,
    ):
        self.update_cache()
        if verbose:
            self.model.summary()
        n_train_steps = len(self.stale_dataset)
        n_val_steps = len(self._validation_ds)
        params = dict(epochs=epochs, verbose=verbose, steps=n_train_steps)

        cb = tf.keras.callbacks.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=True,
            model=self.stale_model,
            **params,
        )
        del callbacks

        train_step = self.stale_model.make_train_function()
        val_step = self._output_model.make_test_function()

        train_iter = iter(self.stale_dataset.repeat())
        val_iter = iter(self._validation_ds.repeat())

        self.stale_model.stop_training = False
        cb.on_train_begin(logs=None)
        initial_epoch = self.stale_model._maybe_load_initial_epoch_from_ckpt(  # pylint: disable=protected-access
            initial_epoch
        )

        logs = None

        for epoch in range(initial_epoch, epochs):
            self.stale_model.reset_metrics()
            cb.on_epoch_begin(epoch, logs=logs)
            for batch in range(n_train_steps):
                cb.on_train_batch_begin(batch=batch)
                logs = train_step(train_iter)
                cb.on_train_batch_end(batch=batch, logs=logs)
                if self.stale_model.stop_training:
                    break
            # print("\nUpdating cache...")
            self.update_cache()
            # print("  Finished")
            self._output_model.reset_metrics()
            for batch in range(n_val_steps):
                cb.on_test_batch_begin(batch)
                val_logs = val_step(val_iter)
                cb.on_test_batch_end(batch, val_logs)
            logs.update({f"val_{k}": v for k, v in val_logs.items()})
            cb.on_epoch_end(epoch, logs)
            if self.stale_model.stop_training:
                break

        cb.on_train_end(logs)

        return self.stale_model.history

    def _test(self, dataset: tf.data.Dataset):
        self.update_cache()
        return self._output_model.evaluate(dataset, verbose=False, return_dict=True)

    def test(self):
        return self._test(self._test_ds)

    def validate(self):
        return self._test(self._validation_ds)


def _print_results(results: tp.Mapping[str, tp.Any]):
    # print results
    width = max(len(k) for k in results) + 1
    for k in sorted(results):
        print(f"{k.ljust(width)}: {results[k]}")


@configurable
def fit_and_test(
    trainer: StaleTrainer,
    epochs: int,
    verbose: bool = True,
    callbacks: tp.Iterable[tf.keras.callbacks.Callback] = (),
    initial_epoch: int = 0,
) -> tp.Mapping[str, float]:
    history = trainer.fit(
        epochs=epochs, verbose=verbose, callbacks=callbacks, initial_epoch=initial_epoch
    )

    val_results = trainer.validate()
    test_results = trainer.test()

    results = {f"val_{k}": v for k, v in val_results.items()}
    for k, v in test_results.items():
        results[f"test_{k}"] = v

    _print_results(results)
    return trainer, history, results


@register
def fit_and_test_many(repeats: int, seed: int = 0, **kwargs):
    test_results = {}
    results = repeat(
        functools.partial(fit_and_test, **kwargs), repeats=repeats, seed=seed
    )
    for res in results:
        # print(f"Starting run {i+1} / {repeats}")
        for k, v in res[-1].items():
            test_results.setdefault(k, []).append(v)
    print(f"Results for {repeats} runs")
    _print_results({k: f"{np.mean(v)} +- {np.std(v)}" for k, v in test_results.items()})
    return test_results


@register
def get_log_dir(
    batch_size: int, variant: str, problem: str, gtf_dir: str = "$GTF_DATA_DIR"
) -> str:
    path = os.path.join(gtf_dir, "stale_gcn", variant, problem, f"b{batch_size:06d}")
    return os.path.expanduser(os.path.expandvars(path))

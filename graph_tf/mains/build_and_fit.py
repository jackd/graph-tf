import functools
from typing import Any, Callable, Dict, Iterable, Tuple

import gin
import numpy as np
import tensorflow as tf

from graph_tf.data.types import DataSplit
from graph_tf.utils.train import fit, unpack
from graph_tf.utils.type_specs import get_type_spec


class InfiniteDataset:
    def __init__(self, dataset: tf.data.Dataset, steps_per_epoch: int):
        assert isinstance(dataset, tf.data.Dataset)
        assert len(dataset) == tf.data.INFINITE_CARDINALITY
        self._dataset = dataset
        self._steps_per_epoch = steps_per_epoch

    @property
    def dataset(self):
        return self._dataset

    @property
    def steps_per_epoch(self):
        return self._steps_per_epoch


def first(iterable: Iterable):
    it = iter(iterable)
    try:
        return next(it)
    except StopIteration as e:
        raise ValueError("iterable must have at least one entry") from e


@gin.configurable(module="gtf")
def build_and_fit(
    data: DataSplit,
    model_fn: Callable[[Any], tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics=None,
    weighted_metrics=None,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    epochs: int = 1,
    validation_freq: int = 1,
    initial_epoch: int = 0,
    verbose: bool = True,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Any]]:
    train_data, validation_data, test_data = data

    if isinstance(train_data, InfiniteDataset):
        steps_per_epoch = train_data.steps_per_epoch
        train_data = train_data.dataset
    else:
        steps_per_epoch = None

    if isinstance(train_data, tf.data.Dataset):
        spec = train_data.element_spec[0]
    else:
        spec = get_type_spec(first(train_data)[0])

    model: tf.keras.Model = model_fn(spec)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )
    if verbose:
        model.summary()
    history = fit(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_freq=validation_freq,
        callbacks=callbacks,
        verbose=verbose,
        steps_per_epoch=steps_per_epoch,
    )

    results = {}

    def evaluate(data):
        if isinstance(data, tf.data.Dataset):
            verbose = len(data) > 1
        else:
            verbose = False
            data = tf.data.Dataset.from_tensors(unpack(data))
        return model.evaluate(data, return_dict=True, verbose=verbose)

    def print_results(results):
        width = max(len(k) for k in results) + 1
        for k in sorted(results):
            print(f"{k.ljust(width)}: {results[k]}")

    if validation_data is not None:
        val_res = evaluate(validation_data)
        results.update({f"val_{k}": v for k, v in val_res.items()})
    if test_data is not None:
        test_res = evaluate(test_data)
        results.update({f"test_{k}": v for k, v in test_res.items()})
    print("Final results")
    print_results(results)
    return model, history, results


@gin.configurable(module="gtf")
def repeat(fn, repeats: int, seed: int = 0):
    rng = tf.random.Generator.from_seed(seed)
    seeds = tf.unstack(rng.uniform_full_int((repeats, 2)), axis=0)
    results = []
    for s in seeds:
        s0, s1 = s.numpy()
        tf.random.set_seed(s0)
        tf.random.get_global_generator().reset_from_seed(s1)
        results.append(fn())
    return results


@gin.configurable(module="gtf")
def build_and_fit_many(repeats: int, seed: int = 0, **kwargs):
    test_results = {}
    results = repeat(
        functools.partial(build_and_fit, **kwargs), repeats=repeats, seed=seed
    )
    for res in results:
        # print(f"Starting run {i+1} / {repeats}")
        for k, v in res[-1].items():
            test_results.setdefault(k, []).append(v)
    print(f"Results for {repeats} runs")
    width = max(len(k) for k in test_results)
    for k in sorted(test_results):
        v = test_results[k]
        print(f"{k.ljust(width)} = {np.mean(v)} +- {np.std(v)}")
    return test_results

import functools
from typing import Any, Callable, Dict, Iterable, Tuple

import gin
import numpy as np
import tensorflow as tf

from graph_tf.utils.train import fit
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


@gin.configurable(module="gtf")
def build_and_fit(
    data,
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
    if len(data) == 1:
        (train_data,) = data
        validation_data = None
        test_data = None
    elif len(data) == 2:
        train_data, validation_data = data
        test_data = None
    else:
        assert len(data) == 3
        train_data, validation_data, test_data = data

    if isinstance(train_data, InfiniteDataset):
        steps_per_epoch = train_data.steps_per_epoch
        train_data = train_data.dataset
    else:
        steps_per_epoch = None

    if isinstance(train_data, tf.data.Dataset):
        spec = train_data.element_spec[0]
    else:
        spec = get_type_spec(train_data[0])

    model = model_fn(spec)
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
    if test_data is not None:
        if isinstance(test_data, tf.data.Dataset):
            verbose = len(test_data) > 1
        else:
            verbose = False
            test_data = tf.data.Dataset.from_tensors(test_data)
        test_res = model.evaluate(test_data, return_dict=True, verbose=verbose)
        print("Results on test data:")
        width = max(len(k) for k in test_res.keys()) + 1
        for k in sorted(test_res):
            print(f"{k.ljust(width)}: {test_res[k]}")
    else:
        test_res = None
    return model, history, test_res


def repeat(fn, repeats: int, seed: int = 0):
    rng = tf.random.Generator.from_seed(seed)
    seeds = tf.unstack(rng.uniform_full_int((repeats, 3)), axis=0)
    results = []
    for s in seeds:
        s0, s1, s2 = s.numpy()
        tf.random.set_seed(s0)
        tf.random.get_global_generator().reset_from_seed([s1, s2])
        yield fn()
    return results


@gin.configurable(module="gtf")
def build_and_fit_many(repeats: int, seed: int = 0, **kwargs):
    test_results = {}
    for i, res in enumerate(
        repeat(functools.partial(build_and_fit, **kwargs), repeats=repeats, seed=seed)
    ):
        print(f"Starting run {i+1} / {repeats}")
        for k, v in res[-1].items():
            test_results.setdefault(k, []).append(v)
    print(f"Results for {repeats} runs")
    width = max(len(k) for k in test_results)
    for k in sorted(test_results):
        v = test_results[k]
        print(f"{k.ljust(width)} = {np.mean(v)} +- {np.std(v)}")
    return test_results

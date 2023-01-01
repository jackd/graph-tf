import functools
import typing as tp

import gin
import numpy as np
import tensorflow as tf
from gacl import Callback

from graph_tf.utils.train import print_result_stats, print_results

register = functools.partial(gin.register, module="gtf.utils.experiment_callbacks")


@register
class FitReporter(Callback):
    def __init__(self):
        self.metrics = []

    def on_trial_completed(self, trial_id: int, result):
        if hasattr(result, "items"):
            metrics = result
        else:
            model, history, metrics = result
            del model, history
        print(f"Completed trial {trial_id}")
        print_results(metrics)
        self.metrics.append(metrics)

    def on_end(self):
        if len(self.metrics) > 1:
            print(f"Completed {len(self.metrics)} trials")
            print_result_stats(self.metrics)


@register
class NumpySeedSetter(Callback):
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.seeds = None

    def on_start(self, num_trials: tp.Optional[int]):
        assert num_trials is not None
        rng = np.random.default_rng(self.seed)
        self.seeds = rng.integers(0, high=np.iinfo(np.int32).max, size=num_trials)

    def on_trial_start(self, trial_id: int):
        np.random.seed(self.seeds[trial_id])


@register
class TensorflowSeedSetter(Callback):
    def __init__(self, seed: int = 0, alg=None):
        self.seed = seed
        self.alg = alg
        self.seeds = None

    def on_start(self, num_trials: tp.Optional[int]):
        assert num_trials is not None
        rng = tf.random.Generator.from_seed(self.seed, self.alg)
        self.seeds = rng.uniform_full_int(shape=(num_trials,)).numpy()

    def on_trial_start(self, trial_id: int):
        tf.random.set_seed(self.seeds[trial_id])


@register
class TensorflowRngSetter(Callback):
    def __init__(self, seed: int = 0, alg=None):
        self.seed = seed
        self.alg = alg
        self.rngs = None

    def on_start(self, num_trials: tp.Optional[int]):
        assert num_trials is not None
        self.rngs = tf.random.Generator.from_seed(self.seed, self.alg).split(num_trials)

    def on_trial_start(self, trial_id: int):
        tf.random.set_global_generator(self.rngs[trial_id])


@register
class TensorflowDeterministicConfigurer(Callback):
    def on_start(self, num_trials: tp.Optional[int]):
        del num_trials
        tf.config.experimental.enable_op_determinism()

import functools

import gin
import numpy as np
import tensorflow as tf

from gacl import Callback

register = functools.partial(gin.register, module="gtf.utils.experiment_callbacks")


@register
class NumpySeedSetter(Callback):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def on_start(self):
        np.random.seed(self.seed)


@register
class TensorflowSeedSetter(Callback):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def on_start(self):
        tf.random.set_seed(self.seed)


@register
class TensorflowRngSetter(Callback):
    def __init__(self, seed: int = 0, alg=None):
        self.seed = seed
        self.alg = alg

    def on_start(self):
        tf.random.set_global_generator(
            tf.random.Generator.from_seed(self.seed, self.alg)
        )

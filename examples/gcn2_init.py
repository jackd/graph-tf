import numpy as np
import tensorflow as tf


class GCN2KernelInitializer(tf.keras.initializers.Initializer):
    def __init__(self, seed=None):
        self.seed = seed
        super().__init__()

    def get_config(self):
        config = super().get_config()
        config["seed"] = self.seed
        return config

    def __call__(self, shape, dtype=None, **kwargs):
        assert not kwargs
        if dtype is None:
            dtype = tf.float32
        _, features_out = shape
        std = 1 / np.sqrt(float(features_out))
        # raise Exception(std)
        return tf.keras.initializers.RandomUniform(-std, std, seed=self.seed)(
            shape, dtype
        )


i0 = GCN2KernelInitializer()
i1 = tf.keras.initializers.VarianceScaling(
    scale=1 / 3, mode="fan_out", distribution="uniform"
)

shape = (15, 63)

v0 = i0(shape).numpy().flatten()
v1 = i1(shape).numpy().flatten()
print(v0.min(), v0.max())
print(v1.min(), v1.max())

# import matplotlib.pyplot as plt

# fig, (ax0, ax1) = plt.subplots(1, 2)
# ax0.hist(v0)
# ax1.hist(v1)
# plt.show()

import typing as tp

import h5py
import numpy as np
import tensorflow as tf

Features = tp.Union[np.ndarray, h5py.Dataset]
FeaturesTransform = tp.Union[
    tp.Callable[[tf.Tensor], tf.Tensor],
    tp.Sequence[tp.Callable[[tf.Tensor], tf.Tensor]],
]
Transition = tp.Callable[[np.ndarray], np.ndarray]

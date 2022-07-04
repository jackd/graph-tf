import sys
from typing import Iterable, Optional

import numpy as np
import tensorflow as tf


def _as_tensor(x):
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x)
    return x


def _build_train_step(model, data, jit_compile: bool):
    data = tf.nest.map_structure(_as_tensor, data)

    @tf.function(jit_compile=jit_compile)
    def train_fn():
        return model.train_step(data)

    return train_fn


def _build_test_step(model, data, jit_compile: bool):
    data = tf.nest.map_structure(_as_tensor, data)

    @tf.function(jit_compile=jit_compile)
    def test_fn():
        model.reset_metrics()
        return model.test_step(data)

    return test_fn


class EpochProgbarLogger(tf.keras.callbacks.Callback):
    """Progress bar that updates at the end of each epoch."""

    def __init__(self):
        super().__init__()
        self.progbar = None
        self.epochs = None
        self.last_seen = None

    def set_params(self, params):
        self.epochs = params["epochs"]

    def on_train_begin(self, logs=None):
        del logs

        class Universe:
            """Contains everything."""

            def __contains__(self, x):
                return True

        self.progbar = tf.keras.utils.Progbar(
            target=self.epochs,
            unit_name="epoch",
        )
        # probar uses stateful metrics to determine which metric values to average.
        # Since this is only called on_epoch_end, no metrics should be averaged
        # i.e. all metrics should be considered 'stateful'.
        # don't set stateful_metrics in constructor because it wraps it in `set`.
        self.progbar.stateful_metrics = Universe()

    def on_epoch_end(self, epoch: int, logs=None):
        self.last_seen = epoch + 1
        self.progbar.update(epoch + 1, list(logs.items()))

    def on_train_end(self, logs=None):
        del logs
        if self.last_seen < self.progbar.target:
            if tf.version.VERSION < "2.3":
                sys.stdout.write("\n")
            else:
                self.progbar.update(self.last_seen, finalize=True)


def unpack(data):
    if data is None:
        return data
    if isinstance(data, tf.data.Dataset):
        return data.get_single_element()
    if len(data) == 1:
        return data[0]
    return data


def fit_single(
    model: tf.keras.Model,
    train_data,
    validation_data=None,
    epochs: int = 1,
    initial_epoch: int = 0,
    validation_freq: int = 1,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    verbose: bool = True,
    jit_compile: bool = False,
):
    """
    Optimized keras.Model.fit for training on a single graph.

    Args:
        model: keras model to train.
        train_data: (inputs, labels, sample_weight) or dataset with a
            single element for training.
        validation_data: (inputs, labels, sample_weight) or dataset with a
            single element for validation.
        epochs: int, maximum number of epochs / steps to train for.
        initial_epoch: int, starting epoch.
        validation_freq: int, number of training steps/epochs per validation.
        callbacks: Iterable of tf.keras.callbacks.Callbacks.
        verbose: flag resulting in verbose outputs.
        jit_compile: flag indicating whether train/validation steps are compiled
            with `jit`. Not all ops are jit compatible, though where they are this may
            result in speed-ups.

    Returns:
        history: `tf.keras.callbacks.History` object.
    """
    train_data = unpack(train_data)
    validation_data = unpack(validation_data)
    do_validation = validation_data is not None

    params = dict(
        epochs=epochs,
        verbose=verbose,
        steps=1,
        do_validation=do_validation,
    )
    callbacks = list(callbacks)
    if verbose:
        callbacks.append(EpochProgbarLogger())

    cb = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=False,
        model=model,
        **params,
    )
    del callbacks
    train_step = _build_train_step(model, train_data, jit_compile=jit_compile)
    if validation_data is None:
        validation_step = None
    else:
        validation_step = _build_test_step(
            model, validation_data, jit_compile=jit_compile
        )

    model.stop_training = False
    cb.on_train_begin(logs=None)
    # _maybe_load_initial_epoch_from_ckpt behaviour is influenced by
    # callbacks.experimental.BackupAndRestore
    initial_epoch = (
        model._maybe_load_initial_epoch_from_ckpt(  # pylint: disable=protected-access
            initial_epoch
        )
    )

    logs = None
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        cb.on_epoch_begin(epoch, logs=None)
        cb.on_train_batch_begin(batch=0)
        logs = train_step()
        cb.on_train_batch_end(batch=0, logs=logs)
        if model.stop_training:
            break
        # validation
        if validation_step is not None and (epoch + 1) % validation_freq == 0:
            val_logs = validation_step()
            logs.update({f"val_{k}": v for k, v in val_logs.items()})
        cb.on_epoch_end(epoch, logs)
        if model.stop_training:
            break

    cb.on_train_end(logs)
    return model.history


def fit(
    model: tf.keras.Model,
    train_data,
    validation_data=None,
    epochs: int = 1,
    initial_epoch: int = 0,
    validation_freq: int = 1,
    callbacks: Iterable[tf.keras.callbacks.Callback] = (),
    steps_per_epoch: Optional[int] = None,
    verbose: bool = True,
    jit_compile: bool = False,
):
    """
    Call `fit_single` or `Model.fit` based on `train_data`.

    Delegates to either `graph_tf.train.fit_single` or `tf.keras.Model.fit`.

    Args:
        model: keras model to train.
        train_data: (inputs, labels, sample_weight) or dataset with a
            single element for training.
        validation_data: (inputs, labels, sample_weight) or dataset with a
            single element for validation.
        epochs: int, maximum number of steps/epochs to train for.
        initial_epoch: int, starting epoch.
        validation_freq: int, number of training steps/epochs per validation.
        callbacks: Iterable of `tf.keras.callbacks.Callbacks`.
        steps_per_epoch: Number of steps per epoch. Must be 1 if specified and
            train_data is a not a `tf.data.Dataset`.
        verbose: flag resulting in verbose outputs.
        jit_compile: used in fit_single. Ignored if more than one example.

    Returns:
        history: `tf.keras.callbacks.History` object.
    """
    if not isinstance(train_data, tf.data.Dataset) or train_data.cardinality() == 1:
        assert steps_per_epoch is None or steps_per_epoch == 1
        return fit_single(
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_freq=validation_freq,
            callbacks=callbacks,
            verbose=verbose,
            jit_compile=jit_compile,
        )
    return model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_freq=validation_freq,
        callbacks=callbacks,
        verbose=verbose,
        steps_per_epoch=steps_per_epoch,
    )

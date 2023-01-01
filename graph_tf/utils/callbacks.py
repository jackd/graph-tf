import functools
from time import time

import gin
import numpy as np
import tensorflow as tf

register = functools.partial(gin.register, module="gtf.utils.callbacks")


@register
class MeanStepTimer(tf.keras.callbacks.Callback):
    def __init__(self, start_step: int, stop_step: int, **kwargs):
        super().__init__(**kwargs)
        self.start_step = start_step
        self.stop_step = stop_step
        self.step = 0
        self.start_time = None
        assert start_step < stop_step, (start_step, stop_step)

    def on_train_batch_begin(self, batch, logs=None):
        del batch, logs
        if self.step == self.start_step:
            self.start_time = time()
        self.step += 1

    def on_train_batch_end(self, batch, logs=None):
        del batch, logs
        if self.step == self.stop_step - 1:
            dt = time() - self.start_time
            print("---")
            print(f"Mean time: {dt / (self.stop_step - self.start_step)}")
            print("---")


@register
class StepTimer(tf.keras.callbacks.Callback):
    def __init__(self, skip: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.skip = skip
        self.train_metric = tf.keras.metrics.Mean()
        self.test_metric = tf.keras.metrics.Mean()
        self.train_start_time = None
        self.test_start_time = None
        self.count = 0
        self._supports_tf_logs = True

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.count = 0
        self.train_metric.reset_state()

    def on_test_begin(self, logs=None):
        self.test_metric.reset_state()

    def on_train_batch_begin(self, batch, logs=None):
        super().on_train_batch_begin(batch, logs)
        self.count += 1
        if self.count >= self.skip:
            self.train_start_time = time()

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if self.count >= self.skip:
            end_time = time()
            dt = end_time - self.train_start_time
            # print()
            # print(int(self.start_time * 1000), int(end_time * 1000), int(dt * 1000))
            # print()
            self.train_metric.update_state(dt)
            logs["time"] = self.train_metric.result()
        else:
            logs["time"] = np.nan

    def on_test_batch_begin(self, batch, logs=None):
        super().on_test_batch_begin(batch, logs)
        if self.count >= self.skip:
            self.test_start_time = time()

    def on_test_batch_end(self, batch, logs=None):
        super().on_test_batch_end(batch, logs)
        if self.count >= self.skip:
            self.test_metric.update_state(time() - self.test_start_time)
            logs["time"] = self.test_metric.result()
        else:
            logs["time"] = np.nan


@register
class EarlyStoppingV2(tf.keras.callbacks.EarlyStopping):
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if (
            self.restore_best_weights
            and self.stopped_epoch == 0
            and self.best_weights is not None
        ):
            self.model.set_weights(self.best_weights)


# @register
# class TensorBoardWithActivations(tf.keras.callbacks.TensorBoard):
#     def __init__(self, *args, activation_freq: int = 1, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.activation_freq = activation_freq

#     def on_train_batch_end(self, batch, logs=None):
#         super().on_train_batch_end(batch, logs=None)
#         if self.activation_freq and batch % self.activation_freq == 0:
#             self._log_train_activations()

#     def on_test_batch_end(self, batch, logs=None):
#         super().on_test_batch_end(batch, logs)
#         if self.activation_freq and batch % self.activation_freq == 0:
#             self._log_val_activations()


# @register
# class TensorBoardFix(tf.keras.callbacks.TensorBoard):
#     """Fixes incorrect step values when using custom summary ops."""

#     def __init__(self, *args, **kwargs):
#         kwargs["write_steps_per_second"] = True
#         super().__init__(*args, **kwargs)

#     def on_train_begin(self, *args, **kwargs):
#         super().on_train_begin(*args, **kwargs)
#         # tf.summary.experimental.set_step(self._train_step)
#         tf.summary.experimental.set_step(tf.convert_to_tensor(-2))

#     def on_test_begin(self, *args, **kwargs):
#         super().on_test_begin(*args, **kwargs)
#         # tf.summary.experimental.set_step(self._val_step)
#         tf.summary.experimental.set_step(tf.convert_to_tensor(-3))

#     def on_train_batch_begin(self, batch, logs=None):
#         print("on_train_batch_begin", tf.summary.experimental.get_step().numpy())


# @register
# class ActivationHistogramCallback(tf.keras.callbacks.Callback):
#     """Output activation histograms."""

#     def __init__(self, log_dir: str):
#         """Initialize layer data."""
#         super().__init__()
#         self.writer = tf.summary.create_file_writer(log_dir)
#         self.step = tf.Variable(0, dtype=tf.int64)

#     def set_model(self, _model):
#         """Wrap layer calls to access layer activations."""
#         for layer in self.layers:
#             self.batch_layer_outputs[layer] = tf_nan(layer.output.dtype)

#             def outer_call(inputs, layer=layer, layer_call=layer.call):
#                 outputs = layer_call(inputs)
#                 self.batch_layer_outputs[layer].assign(outputs)
#                 return outputs

#             layer.call = outer_call

#     def on_train_batch_end(self, _batch, _logs=None):
#         """Write training batch histograms."""
#         with self.writer.as_default():
#             for layer, outputs in self.batch_layer_outputs.items():
#                 if isinstance(layer, keras.layers.InputLayer):
#                     continue
#                 tf.summary.histogram(f"{layer.name}/output", outputs, step=self.step)

#         self.step.assign_add(1)

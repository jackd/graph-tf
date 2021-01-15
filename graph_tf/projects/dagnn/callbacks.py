import gin
import numpy as np
import tensorflow as tf


@gin.configurable(module="gtf.dagnn")
class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience: int, monitor="val_loss"):
        self.patience = patience
        self.monitor = monitor
        self.best_weights = None
        self.best = np.inf
        self.epochs = None
        self.model = None
        super().__init__()

    def on_train_begin(self, logs=None):
        self.best_weights = None
        self.best = np.inf

    def set_params(self, params):
        self.epochs = params["epochs"]

    def on_epoch_end(self, epoch: int, logs=None):
        loss = logs[self.monitor]
        if loss < self.best:
            self.best_weights = self.model.get_weights()
            self.best = loss

        if epoch > self.epochs // 2:
            losses = self.model.history.history[self.monitor]
            if loss > np.mean(losses[-(self.patience + 1) : -1]):
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

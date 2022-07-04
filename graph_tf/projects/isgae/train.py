import functools
import os
import typing as tp

import gin
import tensorflow as tf

from graph_tf.data.single import AutoencoderData, DataSplit
from graph_tf.mains.build_and_fit import build_and_fit

register = functools.partial(gin.register, module="gtf.isgae")


def f(A, X):
    D = tf.expand_dims(tf.sparse.reduce_sum(A, axis=1), axis=1)
    AX = tf.sparse.sparse_dense_matmul(A, X) / D
    return tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        (X, AX), axis=1
    )


def build_and_fit_iteration(
    data: AutoencoderData,
    encoder_fn: tp.Callable[[tf.TensorSpec], tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics=None,
    weighted_metrics=None,
    callbacks: tp.Iterable[tf.keras.callbacks.Callback] = (),
    epochs: int = 1,
    validation_freq: int = 1,
    verbose: bool = True,
):
    X = f(data.adjacency, data.features)

    def get_examples(labels, weights):
        example = (X, labels, weights)
        return (example,)

    data_split = DataSplit(
        get_examples(data.train_labels, data.train_weights),
        get_examples(data.true_labels, data.validation_weights),
        get_examples(data.true_labels, data.test_weights),
    )
    encoder = encoder_fn(tf.TensorSpec(X.shape, X.dtype))

    def model_fn(spec):
        inp = tf.keras.Input(type_spec=spec)
        embedding = encoder(inp)
        preds = tf.matmul(embedding, embedding, transpose_b=True)
        # preds = tf.nn.sigmoid(preds)
        output = tf.reshape(preds, (-1, 1))
        return tf.keras.Model(inp, output)

    _, history, test_result = build_and_fit(
        data_split,
        model_fn,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
        callbacks=callbacks,
        epochs=epochs,
        validation_freq=validation_freq,
        verbose=verbose,
    )
    embedding = encoder(X)
    return embedding, history, test_result


@register
def iterative_build_and_fit(
    data: AutoencoderData,
    encoder_fn: tp.Callable[[tf.TensorSpec], tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics=None,
    weighted_metrics=None,
    callbacks: tp.Iterable[tf.keras.callbacks.Callback] = (),
    epochs: int = 1,
    validation_freq: int = 1,
    verbose: bool = True,
    iterations: int = 3,
    log_dir: tp.Optional[str] = None,
):

    callbacks = tuple(callbacks)
    test_results = []
    for i in range(iterations):
        if log_dir is None:
            cbs = callbacks
        else:
            cbs = list(callbacks)
            cbs.append(
                tf.keras.callbacks.TensorBoard(os.path.join(log_dir, f"iter-{i:03d}"))
            )
        embedding, history, test_result = build_and_fit_iteration(
            data=data,
            encoder_fn=encoder_fn,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            callbacks=cbs,
            epochs=epochs,
            validation_freq=validation_freq,
            verbose=verbose,
        )
        test_results.append(test_result)
        del history
        data = AutoencoderData(
            embedding,
            adjacency=data.adjacency,
            train_labels=data.train_labels,
            true_labels=data.true_labels,
            train_weights=data.train_weights,
            validation_weights=data.validation_weights,
            test_weights=data.test_weights,
        )
    return tuple(test_result)

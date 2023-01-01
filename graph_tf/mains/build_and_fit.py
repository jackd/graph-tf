import functools
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import gin
import tensorflow as tf

from graph_tf.data.data_types import DataSplit
from graph_tf.utils.train import finalize, fit
from graph_tf.utils.type_specs import get_type_spec

register = functools.partial(gin.register, module="gtf")


def first(iterable: Iterable):
    it = iter(iterable)
    try:
        return next(it)
    except StopIteration as e:
        raise ValueError("iterable must have at least one entry") from e


@register
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
    steps_per_epoch: Optional[int] = None,
    force_normal: bool = False,
    skip_validation: bool = False,
    skip_test: bool = False,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Any]]:
    train_data, validation_data, test_data = data
    # for data in train_data, validation_data, test_data:
    #     l = data.get_single_element()[1]
    #     print(
    #         (
    #             tf.math.unsorted_segment_sum(
    #                 tf.ones_like(l), l, num_segments=tf.reduce_max(l).numpy() + 1
    #             )
    #             / tf.shape(l, l.dtype)[0]
    #         ).numpy()
    #     )
    # raise Exception()
    # tf.summary.experimental.set_step(optimizer.iterations)

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
    if skip_validation:
        validation_data = None
    if skip_test:
        test_data = None
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
        force_normal=force_normal,
    )
    results = finalize(model, validation_data, test_data, callbacks)
    return model, history, results

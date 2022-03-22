"""Keras Tuner configurables."""
import functools
import typing as tp

import gin
import keras
import keras_tuner as kt
from keras_tuner.tuners.sklearn_tuner import SklearnTuner

register = functools.partial(gin.register, module="kt")


@register
def Boolean(
    hp: kt.HyperParameters,
    name: str,
    default: bool = False,
    parent_name: tp.Optional[str] = None,
    parent_values: tp.Optional[tp.Any] = None,
) -> bool:
    return hp.Boolean(
        name=name,
        default=default,
        parent_name=parent_name,
        parent_values=parent_values,
    )


@register
def Choice(
    hp: kt.HyperParameters,
    values: tp.Sequence,
    name: str,
    ordered: tp.Optional[bool] = None,
    default: tp.Optional[tp.Any] = None,
    parent_name: tp.Optional[str] = None,
    parent_values: tp.Optional[tp.Any] = None,
):
    return hp.Choice(
        values=values,
        name=name,
        ordered=ordered,
        default=default,
        parent_name=parent_name,
        parent_values=parent_values,
    )


@register
def Fixed(
    hp: kt.HyperParameters,
    value,
    parent_name: tp.Optional[str] = None,
    parent_values: tp.Optional[tp.Any] = None,
):
    return hp.Fixed(value=value, parent_name=parent_name, parent_values=parent_values)


@register
def Float(
    hp: kt.HyperParameters,
    name: str,
    min_value: float,
    max_value: float,
    step: tp.Optional[float] = None,
    sampling: tp.Optional[str] = None,
    default: tp.Optional[float] = None,
    parent_name: tp.Optional[str] = None,
    parent_values=None,
):
    return hp.Float(
        name=name,
        min_value=min_value,
        max_value=max_value,
        step=step,
        sampling=sampling,
        default=default,
        parent_name=parent_name,
        parent_values=parent_values,
    )


@register
def Int(
    hp: kt.HyperParameters,
    name: str,
    min_value: int,
    max_value: int,
    step: int = 1,
    sampling: tp.Optional[str] = None,
    default: tp.Optional[str] = None,
    parent_name: tp.Optional[str] = None,
    parent_values=None,
):
    return hp.Int(
        name=name,
        min_value=min_value,
        max_value=max_value,
        step=step,
        sampling=sampling,
        default=default,
        parent_name=parent_name,
        parent_values=parent_values,
    )


@register
# pylint: disable=abstract-method
class LambdaHyperModel(kt.HyperModel):
    def __init__(
        self,
        build: tp.Callable[[kt.HyperParameters], keras.Model],
        fit: tp.Optional[tp.Callable[[kt.HyperParameters], tp.Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.build = build
        if fit is not None:
            self.fit = fit


@register
def search(tuner: kt.Tuner, **fit_kwargs):
    tuner.search(**fit_kwargs)


@register
def build_tuner(tuner_cls: type, **kwargs):
    return tuner_cls(**kwargs)


for tuner_cls in (kt.BayesianOptimization, kt.RandomSearch, kt.Hyperband, SklearnTuner):
    register(tuner_cls)

register(kt.Objective)

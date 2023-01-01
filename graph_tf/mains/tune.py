import functools
import typing as tp

import gacl
import gin
import ray
from gacl.callbacks import Callback
from ray import tune
from ray.air.result import Result

register = functools.partial(gin.register, module="gtf.tune")
configurable = functools.partial(gin.configurable, module="gtf.tune")


for fn in (tune.grid_search,):
    gin.register(tune.grid_search, module="tune")


@register
class InnerTuneReporter(Callback):
    def on_trial_completed(self, trial_id: int, result):
        del trial_id
        if hasattr(result, "items"):
            metrics = result
        else:
            model, history, metrics = result
            del model, history
        tune.report(**metrics)


@register
def with_inner_tune_reporter(
    callbacks: tp.Union[Callback, tp.Iterable[Callback]]
) -> tp.List[Callback]:
    if isinstance(callbacks, Callback):
        callbacks = [callbacks]
    else:
        callbacks = list(callbacks)
    callbacks.append(InnerTuneReporter())
    return callbacks


@register
class OuterTuneReporter(Callback):
    def __init__(self, metric: str, mode: str = "min", scope: str = "avg"):
        self.metric = metric
        self.mode = mode
        self.scope = scope

    def on_trial_completed(self, trial_id: int, result: tune.ResultGrid):
        del trial_id
        best_result = result.get_best_result(
            metric=self.metric, mode=self.mode, scope=self.scope
        )
        print("Best config:")
        print(best_result.config)
        print(best_result.metrics)


@configurable
def objective(
    fun: tp.Callable = gin.REQUIRED,
    callbacks: tp.Union[Callback, tp.Iterable[Callback]] = (),
    num_trials: int = 1,
):
    gacl.main(fun, callbacks=callbacks, num_trials=num_trials)


def _tune(
    tune_config: tp.Mapping[str, tp.Any],
    gin_config: str,
):
    bindings = [gin_config, *(f"{k}={v}" for k, v in tune_config.items())]
    gin.clear_config()  # ensure singletons are reset
    with gin.unlock_config():
        gin.parse_config(bindings)
    objective()


@register
def fit(param_space: tp.Mapping) -> Result:
    gin_config = gin.config.config_str()
    ray.init(local_mode=True)
    tuner = tune.Tuner(
        functools.partial(_tune, gin_config=gin_config),
        param_space=param_space,
        # run_config=RunConfig(resources_per_worker={"GPU": 1}),
    )
    return tuner.fit()

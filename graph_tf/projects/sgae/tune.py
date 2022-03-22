import functools
import os
import typing as tp

import gin
import keras_tuner as kt
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp_lib

from graph_tf.data import transforms
from graph_tf.projects.sgae import core
from graph_tf.utils import train as train_lib
from graph_tf.utils.models import mlp

register = functools.partial(gin.register, module="gtf.sgae.tune")


def _compile(hp: kt.HyperParameters, model: tf.keras.Model):
    del hp
    # lr = hp.Float("learning_rate", 1e-3, 1e-1, sampling="log")
    lr = 1e-2
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="sum"),
        weighted_metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc", from_logits=True),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr", from_logits=True),
        ],
    )


@register(denylist=["hp"])
def build_v1(hp: kt.HyperParameters, base_feature_size: int = 0):
    spectral_size = hp.Choice("spectral_size", values=[8, 16, 32, 64], ordered=True)
    dropout_rate = hp.Float("dropout_rate", 0.0, 0.8, step=0.1)
    output_units = hp.Choice("embedding_size", [8, 16, 32, 64, 128], ordered=True)
    hidden_units = hp.Choice(
        "hidden_units", values=[32, 64, 128, 256, 512], ordered=True
    )
    hidden_layers = hp.Int("hidden_layers", min_value=1, max_value=3)
    spec = tf.TensorSpec((None, spectral_size + base_feature_size,), dtype=tf.float32,)
    model = core.sgae(
        spec,
        functools.partial(
            mlp,
            output_units=output_units,
            hidden_units=(hidden_units,) * hidden_layers,
            dropout_rate=dropout_rate,
        ),
    )
    _compile(hp, model)
    return model


@register(denylist=["hp"])
def data_transform_v1(
    hp: kt.HyperParameter,
    data: core.AutoencoderData,
    *,
    use_laplacian: tp.Optional[bool] = None,
) -> core.DataSplit:
    spectral_size: int = hp.get("spectral_size")
    if use_laplacian is None:
        use_laplacian = hp.Boolean("use_laplacian")
    else:
        use_laplacian = hp.Fixed("use_laplacian", use_laplacian)
    if use_laplacian:
        adjacency_transform = transforms.laplacian
        which = "SM"
    else:
        adjacency_transform = [transforms.add_identity, transforms.normalize_symmetric]
        which = "LM"
    return core.get_spectral_split(
        data,
        spectral_size,
        which=which,
        adjacency_transform=adjacency_transform,
        features_transform=[transforms.row_normalize],
    )


def kt_to_hparam(hp: kt.HyperParameter) -> hp_lib.HParam:
    if isinstance(hp, kt.engine.hyperparameters.Float):
        domain = hp_lib.RealInterval(hp.min_value, hp.max_value)
    elif isinstance(hp, kt.engine.hyperparameters.Int):
        domain = hp_lib.IntInterval(hp.min_value, hp.max_value)
    elif isinstance(hp, kt.engine.hyperparameters.Boolean):
        domain = hp_lib.Discrete([False, True], dtype=bool)
    elif isinstance(hp, kt.engine.hyperparameters.Fixed):
        domain = hp_lib.Discrete([hp.value])
    elif isinstance(hp, kt.engine.hyperparameters.Choice):
        domain = hp_lib.Discrete(hp.values)
    else:
        raise TypeError(f"Unsupposed hyperparamter type {hp}")
    return hp_lib.HParam(hp.name, domain)


def hp_to_hparams(hp: kt.HyperParameters) -> tp.Mapping[str, hp_lib.HParam]:
    return {kt_to_hparam(param): hp.values[param.name] for param in hp.space}


@register(denylist=["hp", "model"])
def fit(
    hp: kt.HyperParameters,
    model: tf.keras.Model,
    data_fn: tp.Callable[[kt.HyperParameters], core.DataSplit],
    log_dir: tp.Optional[str] = None,
    **kwargs,
):
    split = data_fn(hp)
    callbacks = list(kwargs.pop("callbacks", []))
    if log_dir is not None:
        run = 0
        values = dict(hp.values)
        lap_str = "lap" if values.pop("use_laplacian") else "adj"
        subdir = (
            "{lap_str}{spectral_size}-{hidden_layers}x{hidden_units}-"
            "{embedding_size}_d{dropout_rate:.1f}"
        ).format(lap_str=lap_str, **values)
        log_dir = os.path.join(log_dir, subdir)

        def full_log_dir(log_dir, run):
            return os.path.join(log_dir, f"run-{run:03d}")

        while os.path.exists(full_log_dir(log_dir, run)):
            run += 1
        log_dir = full_log_dir(log_dir, run)
        hparams = hp_to_hparams(hp)
        callbacks.extend(
            [
                tf.keras.callbacks.TensorBoard(log_dir),
                hp_lib.KerasCallback(log_dir, hparams),
            ]
        )
    history = train_lib.fit(
        model, split.train_data, split.validation_data, callbacks=callbacks, **kwargs
    )
    # don't trust history - might have `EarlyStopping` with `restore_best_weights`
    del history
    metrics = {}
    for prefix, d in (("val", split.validation_data), ("test", split.test_data)):
        if d is not None:
            if not isinstance(d, tf.data.Dataset) and len(d) == 1:
                d = tf.data.Dataset.from_tensors(d[0])
            m = model.evaluate(d, return_dict=True, verbose=False)
            metrics.update({f"{prefix}_{k}": v for k, v in m.items()})

    return metrics

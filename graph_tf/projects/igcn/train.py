import functools
import typing as tp

import gin
import tensorflow as tf
from tflo.matrix import Matrix, SparseMatrix

from graph_tf.data import transforms
from graph_tf.data.single import SemiSupervisedSingle
from graph_tf.projects.igcn import data as data_lib
from graph_tf.utils.train import fit

register = functools.partial(gin.register, module="gtf.igcn.train")


def _get_leading_dense(mlp: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in mlp.layers:
        if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.InputLayer)):
            continue
        if isinstance(layer, tf.keras.layers.Dense):
            return layer
        raise ValueError(f"Leading layers must be Dropout or Dense, got {type(layer)}")
    raise ValueError("mlp did not have any linear layers")


class SparseInputPropagatedEvaluator:
    def __init__(self, mlp: tf.keras.Model):
        assert len(mlp.inputs) == 1
        assert len(mlp.outputs) == 1
        layer = _get_leading_dense(mlp)
        self._layer = layer
        self._model = tf.keras.Model(layer.output, mlp.output)
        self._orig = mlp

    def eval(
        self,
        features: tf.SparseTensor,
        propagator: Matrix,
        labels: tf.Tensor,
        sample_weight: tp.Optional[tf.Tensor] = None,
        batch_size: int = -1,
    ) -> tp.Mapping[str, tf.Tensor]:
        metrics = self._orig.compiled_metrics
        if not metrics.built:
            raise ValueError("No metrics, or model.metrics hasn't been built compiled")

        features = tf.linalg.matmul(features, self._layer.kernel)
        features = propagator @ features
        if self._layer.use_bias:
            features += self._layer.bias
        features = self._layer.activation(features)

        metrics.reset_state()

        if batch_size == -1:
            dataset = tf.data.Dataset.from_tensors((features, labels, sample_weight))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (features, labels, sample_weight)
            ).batch(batch_size)

        @tf.function
        def fn(features, labels, sample_weight=None):
            output = self._model(features)
            # output = self._orig(features)  # HACK
            metrics.update_state(labels, output, sample_weight)
            return output

        for el in dataset:
            fn(*el)
        unweighted = {m.name: m.result() for m in metrics.unweighted_metrics}
        weighted = {m.name: m.result() for m in metrics.weighted_metrics}
        for k in unweighted:
            assert k not in weighted, (k, list(weighted))
        weighted.update(unweighted)
        return weighted


@register
def build_fit_test(
    data: SemiSupervisedSingle,
    epsilon: float,
    model_fn: tp.Callable[[tf.TypeSpec], tf.keras.Model],
    loss: tf.keras.losses.Loss,
    optimizer: tf.keras.optimizers.Optimizer,
    *,
    features_transform=(),
    metrics: tp.Iterable[tf.keras.metrics.Metric] = (),
    weighted_metrics: tp.Iterable[tf.keras.metrics.Metric] = (),
    train_batch_size: int = -1,
    test_batch_size: int = -1,
    tol: float = 1e-5,
    max_iter: int = 20,
    preprocess_train: bool = True,
    preprocess_test: bool = False,
    epochs: int = 1,
    callbacks: tp.Iterable[tf.keras.callbacks.Callback] = (),
    steps_per_epoch: tp.Optional[int] = None,
    verbose: bool = True,
    show_progress: bool = True,
):
    print("Starting build_fit_test")
    features = transforms.transformed(data.node_features, features_transform)

    if isinstance(features, tf.SparseTensor):
        features = SparseMatrix(features)

    laplacian = data_lib.get_shifted_laplacian(data.adjacency, epsilon=epsilon)
    print("Computing training data...")
    train_prop = data_lib.sparse_cg_solver(
        laplacian,
        data.train_ids,
        tol=tol,
        max_iter=max_iter,
        preprocess=preprocess_train,
        show_progress=show_progress,
    )
    train_feats = train_prop @ features
    train_labels = tf.gather(data.labels, data.train_ids)
    print("Building model...")
    mlp = model_fn(tf.type_spec_from_value(train_feats))
    mlp.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )

    if train_batch_size == -1:
        dataset = tf.data.Dataset.from_tensors((train_feats, train_labels))
    else:
        dataset = (
            tf.data.Dataset.from_tensor_slices((train_feats, train_labels))
            .shuffle(train_labels.shape[0])
            .batch(train_batch_size)
        )
    if verbose:
        mlp.summary()
    print("Fitting...")
    history = fit(
        mlp,
        dataset,
        validation_data=None,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=verbose,
    )
    print("Evaluating on test nodes...")
    evaluator = SparseInputPropagatedEvaluator(mlp)
    test_prop = data_lib.sparse_cg_solver(
        laplacian,
        data.test_ids,
        tol=tol,
        max_iter=max_iter,
        preprocess=preprocess_test,
        show_progress=show_progress,
    )
    test_labels = tf.gather(data.labels, data.test_ids)

    test_results = evaluator.eval(
        features, test_prop, test_labels, batch_size=test_batch_size
    )
    return mlp, history, test_results


if __name__ == "__main__":
    import sys

    from graph_tf.data.single import get_data, with_random_split_ids
    from graph_tf.utils.models import dense, mlp

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    tf.random.set_seed(seed)

    # name = "cora"
    # sparse_features = True
    # num_classes = 7
    # epochs = 100
    # weight_decay = 5e-3
    # feature_transforms = [
    #     transforms.row_normalize,
    #     functools.partial(transforms.to_format, fmt="sparse"),
    # ]
    # hidden_units = (128,)
    # dropout_rate = 0.5
    # epsilon = 0.1
    # train_batch_size = -1
    # test_batch_size = -1
    # random_splits = False

    name = "bojchevski-mag-coarse"
    sparse_features = True
    num_classes = 8
    epochs = 200
    weight_decay = 1e-4
    feature_transforms = ()
    hidden_units = (32,)
    dropout_rate = 0.1
    epsilon = 0.25
    train_batch_size = -1
    test_batch_size = 512
    random_splits = True
    # seed, result
    # 0, {'acc': 0.7680304, 'cross_entropy': 1.1870477}
    # 1, {'acc': 0.7457247, 'cross_entropy': 1.1971984}
    # 2, {'acc': 0.76893574, 'cross_entropy': 1.0733341}
    # 3, {'acc': 0.7546529, 'cross_entropy': 1.301297}
    # 4, {'acc': 0.69451934, 'cross_entropy': 1.5927169}
    # 5, {'acc': 0.7482539, 'cross_entropy': 1.3115343}
    # 6, {'acc': 0.72400796, 'cross_entropy': 1.5631235}
    # 7, {'acc': 0.7387255, 'cross_entropy': 1.2702699}
    # 8, {'acc': 0.74978274, 'cross_entropy': 1.3156145}
    # 9, {'acc': 0.7307407, 'cross_entropy': 1.4844735}
    tol = 1e-3
    max_iter = 100

    dense_fn = functools.partial(
        dense, kernel_regularizer=tf.keras.regularizers.L2(weight_decay)
    )
    model_fn = functools.partial(
        mlp,
        output_units=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        dense_fn=dense_fn,
    )

    with tf.device("/cpu:0"):
        data = get_data(name, sparse_features=sparse_features)
    if random_splits:
        data = with_random_split_ids(data, 20, 200, balanced=False, seed=0)
    _, _, test_result = build_fit_test(
        data,
        epsilon=epsilon,
        model_fn=model_fn,
        features_transform=feature_transforms,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-2),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseCategoricalCrossentropy(
                from_logits=True, name="cross_entropy"
            ),
        ],
        epochs=epochs,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        tol=tol,
        max_iter=max_iter,
    )
    print(f"seed = {seed}")
    print({k: v.numpy() for k, v in test_result.items()})

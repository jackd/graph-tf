import functools
import os
import typing as tp

import gin
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from graph_tf.data.single import SemiSupervisedSingle
from graph_tf.mains.build_and_fit import print_results
from graph_tf.projects.igcn.scalable import eval as eval_lib
from graph_tf.projects.igcn.scalable import train as train_lib
from graph_tf.projects.igcn.scalable.data_types import Features, FeaturesTransform
from graph_tf.utils.train import EpochProgbarLogger

register = functools.partial(gin.register, module="gtf.igcn.scalable")


def assert_symmetric(adj: sp.spmatrix):
    coo = adj.tocsr()
    cooT = adj.transpose().tocsr()
    np.testing.assert_array_equal(coo.data, cooT.data)
    np.testing.assert_array_equal(coo.indices, cooT.indices)
    np.testing.assert_array_equal(coo.indptr, cooT.indptr)
    # np.testing.assert_array_equal(coo.row, cooT.row)
    # np.testing.assert_array_equal(coo.col, cooT.col)
    np.testing.assert_array_equal(coo.shape, cooT.shape)


class FitManager:
    def __init__(
        self,
        train_manager: train_lib.TrainDatasetManager,
        eval_manager: eval_lib.EagerEvaluatorManager,
        adjacency: sp.spmatrix,
        labels: np.ndarray,
        train_ids: np.ndarray,
        validation_ids: np.ndarray,
        test_ids: np.ndarray,
    ):
        assert_symmetric(adjacency)
        self._train_manager = train_manager
        self._eval_manager = eval_manager
        self._adjacency = adjacency
        self._labels = labels
        self._train_ids = train_ids
        self._validation_ids = validation_ids
        self._test_ids = test_ids

    def create_data(
        self,
        block_size: int = 512,
        remove_base_data: bool = True,
    ):
        self._train_manager.create_data(
            adjacency=self._adjacency,
            ids=self._train_ids,
            labels=self._labels[self._train_ids],
            block_size=block_size,
        )
        self._eval_manager.create_data(
            adjacency=self._adjacency,
            ids=self._validation_ids,
            block_size=block_size,
            remove_base_data=remove_base_data,
        )

    @property
    def has_data(self) -> bool:
        return (
            self._train_manager.has_transpose_data
            and self._eval_manager.has_transpose_data
        )

    @property
    def num_classes(self) -> int:
        return self._train_manager.num_classes

    # def build(
    #     self,
    #     mlp_fn: tp.Callable[[tf.TensorSpec, int], tf.keras.Model],
    #     optimizer: tf.keras.optimizers.Optimizer,
    # ) -> tf.keras.Model:
    #     features_spec = tf.TensorSpec((None, self.num_features))
    #     mlp = mlp_fn(features_spec, self.num_classes)
    #     mlp.compile(
    #         optimizer=optimizer, loss=losses.LazyQuadraticCategoricalCrossentropy()
    #     )
    #     return mlp

    def fit(
        self,
        mlp: tf.keras.Model,
        metrics: tp.Sequence[tf.keras.metrics.Metric],
        features: Features,
        batch_size: int,
        epochs: int,
        features_transform: FeaturesTransform,
        temperature: tp.Optional[float],
        val_batch_size: tp.Optional[int] = None,
        steps_per_epoch: int = -1,
        seed: int = 0,
        validation_freq: int = 1,
        verbose: int = 1,
        callbacks: tp.Sequence[tf.keras.callbacks.Callback] = (),
    ):
        if batch_size == -1:
            batch_size = features.shape[0]
        if val_batch_size is None:
            val_batch_size = batch_size
        assert isinstance(mlp, tf.keras.Model)
        if steps_per_epoch == -1:
            steps_per_epoch = features.shape[0] // batch_size
        train_dataset = self._train_manager.get_dataset(
            features,
            batch_size,
            features_transform,
            temperature,
            seed=seed,
        )
        train_dataset = train_dataset.prefetch(-1)

        eval_callback = self._eval_manager.evaluator(
            features,
            self._labels[self._validation_ids],
            val_batch_size,
            features_transform,
        ).callback(metrics, validation_freq=validation_freq)

        callbacks = list(callbacks)
        callbacks.insert(0, eval_callback)
        progbar = (
            EpochProgbarLogger()
            if steps_per_epoch == 1
            else tf.keras.callbacks.ProgbarLogger(count_mode="steps")
        )
        callbacks.insert(1, progbar)
        callbacks = tf.keras.callbacks.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=False,
            model=mlp,
            steps=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
        )

        history = mlp.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    def test(
        self,
        mlp: tf.keras.Model,
        metrics: tp.Sequence[tf.keras.metrics.Metric],
        features: Features,
        batch_size: int,
        features_transform: FeaturesTransform,
        block_size: int = 16,
    ) -> tp.Mapping[str, tp.Any]:
        return eval_lib.lazy_evaluate(
            mlp,
            eval_lib.get_features_dataset(
                features,
                batch_size=batch_size,
                features_transform=features_transform,
            ),
            transition=self._eval_manager.transition(self._adjacency),
            num_nodes=features.shape[0],
            ids=self._test_ids,
            labels=self._labels[self._test_ids],
            metrics=metrics,
            block_size=block_size,
        )


@register
def build_fit_test(
    train_manager: train_lib.TrainDatasetManager,
    eval_manager: eval_lib.EagerEvaluatorManager,
    data: SemiSupervisedSingle,
    mlp_fn: tp.Callable[[tf.TensorSpec, int], tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    metrics: tp.Sequence[tf.keras.metrics.Metric],
    batch_size: int,
    epochs: int,
    val_batch_size: tp.Optional[int] = None,
    features_transform: FeaturesTransform = (),
    temperature: tp.Optional[float] = None,
    callbacks: tp.Sequence[tf.keras.callbacks.Callback] = (),
    seed: int = 0,
    steps_per_epoch: int = -1,
    validation_freq: int = 1,
    verbose: int = 1,
    remove_base_data: bool = True,
):
    adj = data.adjacency
    if isinstance(adj, tf.SparseTensor):
        row, col = adj.indices.numpy().T
        adj = sp.coo_matrix(
            (np.ones(row.shape, dtype=np.float32), (row, col)), shape=adj.shape
        )
    assert sp.issparse(adj)
    adj = (adj + adj.T) / 2
    adj = adj.tocsr()
    assert sp.issparse(adj)
    features = data.node_features
    if isinstance(features, tf.SparseTensor):
        features = tf.sparse.to_dense(features)
    if isinstance(features, tf.Tensor):
        features = features.numpy()
    if isinstance(features, sp.spmatrix):
        features = features.todense()
    assert isinstance(features, np.ndarray)
    # # HACK
    # import tqdm
    # from graph_tf.projects.igcn.scalable.utils import shifted_laplacian_solver

    # transition = shifted_laplacian_solver(adj, epsilon=0.1)
    # smoothed_features = np.zeros_like(features)
    # for i in tqdm.trange(features.shape[1], desc="Propagating input features"):
    #     smoothed_features[:, i] = transition(features[:, i])
    # features = np.concatenate((features, smoothed_features), axis=-1)
    # # end HACK

    manager = FitManager(
        train_manager=train_manager,
        eval_manager=eval_manager,
        adjacency=adj,
        labels=np.array(data.labels),
        train_ids=np.array(data.train_ids),
        validation_ids=np.array(data.validation_ids),
        test_ids=np.array(data.test_ids),
    )

    manager.create_data(remove_base_data=remove_base_data)
    features_spec = tf.TensorSpec((None, features.shape[1]), dtype=np.float32)
    mlp = mlp_fn(features_spec, manager.num_classes)
    mlp.compile(
        loss=train_manager.loss(),
        optimizer=optimizer,
    )
    mlp.summary()

    history = manager.fit(
        mlp,
        metrics=metrics,
        features=features,
        batch_size=batch_size,
        epochs=epochs,
        features_transform=features_transform,
        temperature=temperature,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        validation_freq=validation_freq,
        verbose=verbose,
        callbacks=callbacks,
        val_batch_size=val_batch_size,
    )

    test_result = manager.test(
        mlp=mlp,
        metrics=metrics,
        features=features,
        batch_size=batch_size,
        features_transform=features_transform,
    )
    print("Test results")
    print_results(test_result)
    return history, test_result


if __name__ == "__main__":
    from graph_tf.data.single import get_data
    from graph_tf.utils.models import mlp, prelu

    # from graph_tf.utils.torch_compat import weight_decay_transformer

    tf.random.set_seed(0)
    np.random.seed(0)
    val_batch_size = None

    # problem = "cora"
    # batch_size = -1
    # hidden_units = (64,)
    # dropout_rate = 0.5
    # input_dropout_rate = None
    # normalization = None
    # features_transform = row_normalize

    # problem = "ogbn-arxiv"
    # # batch_size = 8192
    # # batch_size = 16384
    # # batch_size = 16384 * 8
    # hidden_units = (256,) * 3
    # batch_size = -1
    # # hidden_units = (512,) * 2
    # # hidden_units = (512,) * 3
    # normalization = functools.partial(batch_norm, momentum=0.9)
    # # normalization = None
    # dropout_rate = 0.5
    # features_transform = ()
    # steps_per_epoch = 2
    # val_batch_size = 2048
    # input_dropout_rate = 0.0

    problem = "ogbn-arxiv"
    # hidden_units = (128,) * 2  # 52
    hidden_units = (256,) * 1  # 60
    # hidden_units = (256,) * 2  # 37
    # hidden_units = (1024,) * 1  #
    batch_size = -1
    normalization = None
    dropout_rate = 0.5
    features_transform = ()
    # steps_per_epoch = 100
    val_batch_size = 2048
    input_dropout_rate = 0.0
    # input_dropout_rate = None

    # batch_size = 32   # 33
    # batch_size = 64   # 42
    # batch_size = 128  # 46
    # batch_size = 256  # 53
    # batch_size = 512  # 54  # 34 with epsilon=1, 47 @ 1e-2
    # batch_size = 1024  # 58
    # batch_size = 2048  # 58
    # batch_size = 4096  # 58
    # batch_size = 8192  #
    # batch_size = 16384  #
    # batch_size = 32768  # 58
    batch_size = -1  # 52.6
    steps_per_epoch = 10

    # problem = "cora"
    # batch_size = -1
    # hidden_units = (64,)
    # dropout_rate = 0.8
    # normalization = None
    # features_transform = (row_normalize,)
    # steps_per_epoch = -1

    # lr = 5e-3
    lr = 1e-2
    epochs = 1000
    # epsilon = 0.01
    epsilon = 0.1
    # epsilon = 1.0
    temperature = 0.1
    # temperature = None
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # "val_cross_entropy", patience=5, mode="min", restore_best_weights=True
            "val_acc",
            patience=5,
            mode="max",
            restore_best_weights=True,
        ),
        # tf.keras.callbacks.TensorBoard("/tmp/igcn-tb", profile_batch=(10, 20)),
    ]
    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(
            from_logits=True, name="cross_entropy"
        ),
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ]
    data = get_data(problem)
    cache_dir = os.path.join(os.environ["GTF_DATA_DIR"], "igcn/scalable", problem)
    train_dir = os.path.join(cache_dir, "train")
    eval_dir = os.path.join(cache_dir, "eval")
    # train_manager = train_lib.QuadraticTrainDatasetManager(train_dir, epsilon=epsilon)
    # train_manager = train_lib.ProjectedQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, jl_eps=0.05, batch_jl_eps=0.1
    # )  # 0.6874266862869263
    # train_manager = train_lib.ProjectedQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, jl_eps=0.05, batch_jl_eps=0.2
    # )  # 0.682488739490509
    train_manager = train_lib.ProjectedQuadraticTrainDatasetManager(
        train_dir, epsilon=epsilon, jl_eps=0.05, batch_jl_eps=0.5
    )  # 0.6704524159431458
    # train_manager = train_lib.ProjectedQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, jl_eps=0.05
    # )
    # train_manager = train_lib.ProjectedQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, jl_eps=0.1
    # )  # 0.6779828667640686
    # train_manager = train_lib.ProjectedQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, jl_eps=0.2
    # )  # 0.6652470231056213
    # train_manager = train_lib.ProjectedQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, jl_eps=0.5
    # )  # 0.6265662908554077
    # train_manager = train_lib.LowRankQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, k=128
    # )  #  0.5136308670043945
    # train_manager = train_lib.LowRankQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, k=256
    # )  #  0.5916918516159058
    # train_manager = train_lib.LowRankQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, k=512
    # )  #  0.6121021509170532
    # train_manager = train_lib.LowRankQuadraticTrainDatasetManager(
    #     train_dir, epsilon=epsilon, k=1024
    # )  # 0.6250025629997253
    # train_manager = train_lib.DummyTrainDatasetManager(train_dir, jl_eps=0.1)
    # train_manager = train_lib.DummyTrainDatasetManager(train_dir, jl_eps=0.3)
    # train_manager = train_lib.HackyQuadraticTrainDatasetManager(train_dir)
    # train_manager = train_lib.LinearTrainDatasetManager(train_dir)
    eval_manager = eval_lib.EagerEvaluatorManager(eval_dir, epsilon=epsilon)
    optimizer = tf.keras.optimizers.Adam(
        lr,
        # gradient_transformers=[weight_decay_transformer(5e-3)],
    )

    build_fit_test(
        train_manager=train_manager,
        eval_manager=eval_manager,
        data=data,
        mlp_fn=functools.partial(
            mlp,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            input_dropout_rate=input_dropout_rate,
            normalization=normalization,
            activation=prelu,
            # dense_fn=functools.partial(
            #     # dense, kernel_regularizer=tf.keras.regularizers.L2(5e-3)
            #     dense,
            #     # kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            # ),
        ),
        optimizer=optimizer,
        metrics=metrics,
        batch_size=batch_size,
        epochs=epochs,
        features_transform=features_transform,
        temperature=temperature,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        val_batch_size=val_batch_size,
    )

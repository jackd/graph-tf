import functools
import typing as tp

import gin
import tensorflow as tf
from tflo.matrix.extras import GatherMatrix

from graph_tf.data import transforms
from graph_tf.data.single import DataSplit, SemiSupervisedSingle, get_largest_component

register = functools.partial(gin.register, module="gtf.mlp.data")


@register
def preprocess(
    data: SemiSupervisedSingle,
    features_transform: tp.Iterable[tp.Callable] = (),
    adjacency_features: tp.Iterable[tp.Callable] = (),
    dual_features: tp.Iterable[
        tp.Callable[[tf.SparseTensor, tf.Tensor], tf.Tensor]
    ] = (),
    label_features: tp.Iterable[
        tp.Callable[
            [
                tf.SparseTensor,  # adjacency
                tf.Tensor,  # labels
                tf.Tensor,  # train_ids
            ],
            tf.Tensor,
        ]
    ] = (),
    include_transformed_features: bool = True,
    largest_component_only: bool = False,
    normalize_outputs: bool = False,
    device: str = "/cpu:0",
) -> SemiSupervisedSingle:
    adj = data.adjacency
    with tf.device(device):
        if largest_component_only:
            data = get_largest_component(data, directed=False)
        features = data.node_features
        adj = data.adjacency
        features = transforms.transformed(features, features_transform)
        assert isinstance(features, tf.Tensor), features_transform
        features_list = [features] if include_transformed_features else []

        if callable(adjacency_features):
            features_list.append(adjacency_features(adj))
        else:
            features_list.extend([fn(adj) for fn in adjacency_features])
        if callable(dual_features):
            features_list.append(dual_features(adj, features))
        else:
            features_list.extend([fn(adj, features) for fn in dual_features])

        if label_features:
            args = (
                adj,
                tf.gather(data.labels, data.train_ids),
                data.train_ids,
            )
            if callable(label_features):
                features_list.append(label_features(*args))
            else:
                features_list.extend([fn(*args) for fn in label_features])

        if len(features_list) == 0:
            raise RuntimeError("No features")
        if len(features_list) == 1:
            (features,) = features_list
        else:
            features = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                features_list, axis=1
            )
    if normalize_outputs:
        norm = tf.linalg.norm(features, axis=0, keepdims=True)
        features = features / norm
        # for f in features_list:
        #     n = tf.linalg.norm(f, axis=0).numpy()
        #     print(n)
        #     print(n.mean())
        # raise Exception("here")
    return SemiSupervisedSingle(
        features,
        adj,
        data.labels,
        data.train_ids,
        data.validation_ids,
        data.test_ids,
    )


@register
def get_features_split(data: SemiSupervisedSingle, batch_size: int = -1) -> DataSplit:
    def get_split(
        ids: tp.Optional[tf.Tensor], training: bool = False
    ) -> tp.Optional[tf.data.Dataset]:
        if ids is None:
            return None
        inp = tf.gather(data.node_features, ids)
        lab = tf.gather(data.labels, ids)
        weights = tf.fill((ids.shape[0],), 1 / ids.shape[0])
        if batch_size == -1:
            return tf.data.Dataset.from_tensors((inp, lab, weights))
        dataset = tf.data.Dataset.from_tensor_slices((inp, lab, weights))
        if training:
            dataset = dataset.shuffle(len(dataset))
        return dataset.batch(batch_size)

    return DataSplit(
        get_split(data.train_ids, training=True),
        get_split(data.validation_ids),
        get_split(data.test_ids),
    )


@register
def page_rank_preprocess(
    data: SemiSupervisedSingle,
    epsilon: float,
    tol: float = 1e-5,
    max_iter: int = 1000,
    dropout_rate: float = 0.0,
    renormalized: bool = False,
    show_progress: bool = True,
) -> DataSplit:
    features = transforms.to_format(
        transforms.row_normalize(data.node_features), "dense"
    )
    propagator = transforms.page_rank_matrix(
        data.adjacency,
        epsilon=epsilon,
        tol=tol,
        max_iter=max_iter,
        show_progress=show_progress,
        renormalized=renormalized,
    )
    prop_features = propagator @ features

    def get_dataset(ids, dropout_rate=None):
        if ids is None:
            return None
        labels = tf.gather(data.labels, ids)
        weights = tf.fill((ids.shape[0],), 1 / ids.shape[0])
        if dropout_rate:
            prop = GatherMatrix(ids, data.adjacency.shape[0]).to_dense() @ propagator

            def map_fn(features, labels, weights):
                features = prop @ tf.nn.dropout(features, rate=dropout_rate)
                return features, labels, weights

            return tf.data.Dataset.from_tensors(
                (
                    features,
                    labels,
                    weights,
                )
            ).map(map_fn)
        # dropout_rate is Falsy
        return tf.data.Dataset.from_tensors(
            (
                tf.gather(prop_features, ids),
                tf.gather(data.labels, ids),
                weights,
            )
        )

    train_dataset = get_dataset(data.train_ids, dropout_rate)
    val_dataset, test_dataset = (
        get_dataset(ids) for ids in (data.validation_ids, data.test_ids)
    )

    return DataSplit(train_dataset, val_dataset, test_dataset)

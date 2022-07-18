import functools
import typing as tp

import gin
import tensorflow as tf

from graph_tf.data.single import DataSplit, SemiSupervisedSingle, get_largest_component
from graph_tf.data.transforms import transformed

register = functools.partial(gin.register, module="gtf.mlp.data")


@register
def preprocess(
    data: SemiSupervisedSingle,
    features_transform: tp.Iterable[tp.Callable] = (),
    adjacency_features: tp.Iterable[tp.Callable] = (),
    dual_features: tp.Iterable[
        tp.Callable[[tf.SparseTensor, tf.Tensor], tf.Tensor]
    ] = (),
    include_transformed_features: bool = True,
    largest_component_only: bool = False,
    device: str = "/cpu:0",
) -> SemiSupervisedSingle:
    with tf.device(device):
        if largest_component_only:
            data = get_largest_component(data, directed=False)
        features = data.node_features
        adj = data.adjacency
        features = transformed(features, features_transform)
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

        if len(features_list) == 0:
            raise RuntimeError("No features")
        if len(features_list) == 1:
            (features,) = features_list
        else:
            features = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                features_list, axis=1
            )
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

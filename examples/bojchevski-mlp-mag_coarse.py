raise Exception("See graph_tf/projects/igcn/train.py")
# import typing as tp
# import functools
# import tensorflow as tf

# from tflo.matrix.extras import GatherMatrix

# from graph_tf.data.single import (
#     get_data,
#     with_random_split_ids,
#     get_largest_component,
#     SemiSupervisedSingle,
# )
# from graph_tf.utils.models import mlp, dense
# from graph_tf.data.transforms import page_rank_matrix
# from graph_tf.utils.train import fit_single

# seed = 0
# tf.random.set_seed(seed)


# data = get_data("bojchevski-mag-coarse", sparse_features=True)
# num_classes = data.labels.numpy().max() + 1
# data = get_largest_component(data)


# def fit_and_test(
#     data: SemiSupervisedSingle,
#     l2_reg: float = 1e-4,
#     units: tp.Iterable[int] = (32,),
#     dropout_rate: float = 0.1,
#     lr: float = 1e-2,
#     epsilon: float = 0.25,
#     epochs: int = 200,
# ) -> tp.Mapping[str, float]:
#     data = with_random_split_ids(
#         data,
#         train_samples_per_class=20,
#         validation_samples_per_class=200,
#         balanced=False,
#     )

#     input_spec = tf.TensorSpec(
#         tf.TensorSpec(shape=(data.node_features.shape[1]), dtype=tf.float32)
#     )
#     model = mlp(
#         input_spec,
#         num_classes,
#         hidden_units=units,
#         dense_fn=functools.partial(
#             dense, kernel_regularizer=tf.keras.regularizers.L2(l2_reg)
#         ),
#         dropout_rate=dropout_rate,
#     )
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(lr=lr),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
#     )

#     propagator = page_rank_matrix(
#         data.adjacency, epsilon=epsilon, tol=1e-2, max_iter=1000
#     )
#     train_rows = GatherMatrix(data.train_ids, data.adjacency.shape[0])
#     features = tf.linalg.matmul(
#         tf.linalg.matmul(train_rows.to_dense(), propagator), data.node_features
#     )
#     labels = tf.gather(data.labels, data.train_ids)

#     fit_single(model, tf.data.Dataset.from_tensors((features, labels)), epochs=epochs)

#     first_dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)][0]
#     test_model = tf.keras.Model(, model.output)

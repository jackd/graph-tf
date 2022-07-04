# import os
# import typing as tp

# import h5py
# import tensorflow as tf

# from graph_tf.data.single import SemiSupervisedSingle
# from graph_tf.projects.igcn.data import sparse_cg_solver


# class TrainerCache:
#     def __init__(self, group: h5py.Group):
#         self.group = group

#     def initialize(
#         self,
#         coeff_matrix: tf.SparseTensor,
#         features: tf.Tensor,
#         labels: tf.Tensor,
#         train_ids: tf.Tensor,
#         validation_ids: tf.Tensor,
#         test_ids: tf.Tensor,
#         num_classes: int,
#         tol: float = 1e-5,
#         max_iter: int = 20,
#     ):
#         assert tf.executing_eagerly()
#         self.group.attrs["num_nodes"] = coeff_matrix.shape[0]
#         self.group.attrs["num_classes"] = num_classes

#         cm = self.group.require_group("coeff_matrix")
#         cm.require_dataset("indices", data=coeff_matrix.indices.numpy(), exact=True)
#         cm.require_dataset("values", data=coeff_matrix.values.numpy(), exact=True)

#         train_inverse_matrix = sparse_cg_solver(
#             coeff_matrix, train_ids, tol=tol, max_iter=max_iter, preprocess=True
#         )
#         for name, data in (
#             ("features", features),
#             ("labels", labels),
#             ("train_ids", train_ids),
#             ("validation_ids", validation_ids),
#             ("test_ids", test_ids),
#             ("train_inverse_matrix", train_inverse_matrix),
#         ):
#             self.group.require_dataset(name, data=data.numpy(), exact=True)

#     @property
#     def coeff_matrix(self):
#         num_nodes = self.group.attrs["num_nodes"]
#         cm = self.group["coeff_matrix"]
#         return tf.SparseTensor(cm["indices"], cm["values"], (num_nodes, num_nodes))

#     @property
#     def num_nodes(self):
#         return self.group.attrs["num_nodes"]

#     @property
#     def num_classes(self):
#         return self.group.attrs["num_classes"]

#     @property
#     def features(self):
#         return self.group["features"]

#     @property
#     def labels(self):
#         return self.group["labels"]

#     @property
#     def train_ids(self):
#         return self.group["train_ids"]

#     @property
#     def validation_ids(self):
#         return self.group["validation_ds"]

#     @property
#     def test_ids(self):
#         return self.group["test_ids"]

#     def train_dataset(
#         self, examples_per_step: int, labels_per_step: int, seed: int = 0
#     ):
#         dataset = tf.data.Dataset.random(seed).batch(2)
#         train_ids = tf.convert_to_tensor(train_ids)


# def trainer(
#     mlp: tf.keras.Model,
#     coeff_matrix: tf.Tensor,
#     labels: tf.Tensor,
#     label_ids: tf.Tensor,
#     cache: h5py.Group,
# ) -> tp.Tuple[tf.data.Dataset, tp.Callable]:
#     pass


# class Trainer:
#     def __init__(
#         self,
#         mlp: tf.keras.Model,
#         coeff_matrix: tf.Tensor,
#         train_ids: tf.Tensor,
#         validation_ids: tf.Tensor,
#         test_ids: tf.Tensor,
#         steps_per_epoch: int,
#         examples_per_step: int,
#         labels_per_step: tp.Optional[int],
#         cache: h5py.Group,
#         use_quadratic_loss_approx: bool,
#     ):
#         pass

#     def fit_epoch(self, callbacks: tp.Iterable[tf.keras.callbacks.Callback]):
#         if not isinstance(callbacks, tf.keras.callbacks.Callback):
#             callbacks = tf.keras.callbacks.CallbackList(tuple(callbacks))

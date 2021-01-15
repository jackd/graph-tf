import tensorflow as tf

from graph_tf.gat.models import gat
from graph_tf.utils.ops import indices_to_mask
from graph_tf.utils.test_utils import random_sparse


def preprocess_weights(ids: tf.Tensor, num_nodes, normalize: bool = True):
    weights = indices_to_mask(ids, num_nodes, dtype=tf.float32)
    if normalize:
        weights = weights / tf.size(ids, out_type=tf.float32)
    return weights


class GatModelsTest(tf.test.TestCase):
    def test_gather_consistent(self):
        N = 1000
        nf = 8
        num_classes = 5
        nnz = N * 10
        rng = tf.random.Generator.from_seed(0)

        labels = rng.uniform((N,), maxval=num_classes, dtype=tf.int64)
        labels = tf.sort(labels)

        features_spec = tf.TensorSpec((N, nf), tf.float32)
        adj_spec = tf.SparseTensorSpec((N, N), tf.float32)
        ids_spec = tf.TensorSpec((None,), tf.int64)

        features = rng.uniform((N, nf))
        st = random_sparse((N, N), nnz, rng)
        ids = tf.convert_to_tensor([2, 5, 10, 20], tf.int64)
        weights = preprocess_weights(ids, N)

        tf.random.set_seed(0)
        model = gat((features_spec, adj_spec), num_classes)
        preds = model([features, st], training=False)
        v0 = tf.gather(preds, ids, axis=0)
        l0 = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="sum"
        )(labels, preds, weights)

        tf.random.set_seed(0)
        model_gathered = gat((features_spec, adj_spec, ids_spec), num_classes)
        v1 = model_gathered([features, st, ids], training=False)
        l1 = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="sum_over_batch_size"
        )(tf.gather(labels, ids, axis=0), v1)

        self.assertAllClose(v0, v1, rtol=1e-5)
        self.assertAllClose(l0, l1)


if __name__ == "__main__":
    # tf.test.main()
    GatModelsTest().test_gather_consistent()

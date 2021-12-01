from typing import Iterable

import gin
import numpy as np
import tensorflow as tf

from graph_tf.projects.dfnet.cvx_opt import dfnets_coefficients_optimizer
from graph_tf.projects.dfnet.layers import DFNetConv
from graph_tf.utils.ops import normalize_sparse, sparse_negate
from graph_tf.utils.type_specs import keras_input


@gin.configurable(module="gtf.dfnet")
def preprocess_adj(
    indices: tf.Tensor,
    num_nodes: tf.Tensor,
    ka: int = 5,
    kb: int = 3,
    lambda_cut: float = 0.5,
    mu_size: int = 200,
    radius: float = 0.9,
):
    nnz = tf.shape(indices)[0]
    adj = tf.SparseTensor(
        indices, tf.ones((nnz,), dtype=tf.float32), (num_nodes, num_nodes)
    )
    adj = tf.sparse.add(adj, tf.sparse.eye(num_nodes))
    adj = normalize_sparse(adj)
    # very confused about the idea behind this
    # based on
    # L = I - adj
    # L = L - I
    L = sparse_negate(adj)
    L = tf.sparse.to_dense(L).numpy()

    largest_eig = 2  # normalized laplacian
    threshold = largest_eig / 2 - lambda_cut

    def response(x):
        return (x >= threshold).astype(x.dtype)

    # Since the eigenvalues might change, sample eigenvalue domain uniformly.
    mu = np.linspace(0, largest_eig, mu_size)

    # The parameter 'radius' controls the tradeoff between convergence efficiency and
    # approximation accuracy.
    # A higher value of 'radius' can lead to slower convergence but better accuracy.

    b, a, rARMA, error = dfnets_coefficients_optimizer(mu, response, kb, ka, radius)
    del rARMA, error

    h_zero = np.zeros(L.shape[0])

    def L_mult_numerator(coef):
        y = coef.item(0) * np.linalg.matrix_power(L, 0)
        for i in range(1, len(coef)):
            x = np.linalg.matrix_power(L, i)
            y = y + coef.item(i) * x

        return y

    def L_mult_denominator(coef):
        y_d = h_zero
        for i in range(0, len(coef)):
            x_d = np.linalg.matrix_power(L, i + 1)
            y_d = y_d + coef.item(i) * x_d

        return y_d

    poly_num = L_mult_numerator(b)
    poly_denom = L_mult_denominator(a)

    arma_conv_AR = tf.convert_to_tensor(poly_denom, dtype=tf.float32)
    arma_conv_MA = tf.convert_to_tensor(poly_num, dtype=tf.float32)

    return arma_conv_AR, arma_conv_MA


@gin.configurable(module="gtf.dfnet")
def dfnet(
    inputs_spec,
    num_classes: int,
    hidden_l2_reg: float = 9e-2,
    final_l2_reg: float = 1e-10,
    dropout_rate: float = 0.9,
    units: Iterable[int] = (8, 16, 32, 64, 128),
):
    inputs = tf.nest.map_structure(keras_input, inputs_spec)
    if len(inputs) == 3:
        x0, adj, indices = inputs
    else:
        x0, adj = inputs
        indices = None

    arma_conv_AR, arma_conv_MA = adj
    # if indices is not None:
    #     x0 = tf.gather(x0, indices, axis=0)
    #     arma_conv_AR = tf.gather(arma_conv_AR, indices, axis=0)
    #     arma_conv_AR = tf.gather(arma_conv_AR, indices, axis=1)
    #     arma_conv_MA = tf.gather(arma_conv_MA, indices, axis=0)
    #     arma_conv_MA = tf.gather(arma_conv_MA, indices, axis=1)

    x = x0
    for u in units:
        start = x
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = DFNetConv(
            u,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(hidden_l2_reg),
            kernel_constraint=tf.keras.constraints.unit_norm(),
            bias_initializer=tf.keras.initializers.glorot_normal(),
            bias_constraint=tf.keras.constraints.unit_norm(),
            activation="relu",
        )([x, x0, arma_conv_AR, arma_conv_MA])
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        x = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            [start, x], axis=-1
        )

    if indices is not None:
        x = tf.gather(x, indices, axis=0)

    logits = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        kernel_regularizer=tf.keras.regularizers.l2(final_l2_reg),
        kernel_constraint=tf.keras.constraints.unit_norm(),
        # activity_regularizer=tf.keras.regularizers.l2(final_l2_reg),
        bias_initializer=tf.keras.initializers.glorot_normal(),
        bias_constraint=tf.keras.constraints.unit_norm(),
        activation=None,
        name="logits",
    )(x)

    return tf.keras.Model(inputs, logits)

from typing import Union

import tensorflow as tf


def dfnet_conv(
    x: tf.Tensor,
    num_filters: int,
    arma_conv_AR,
    arma_conv_MA,
    input_signal,
    ar_kernel: Union[tf.Tensor, tf.Variable],
    ma_kernel: Union[tf.Tensor, tf.Variable],
):
    # pylint: disable=no-value-for-parameter,redundant-keyword-arg,
    # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
    K = tf.keras.backend

    y1 = K.dot((-arma_conv_AR), x)
    y2 = K.dot(arma_conv_MA, input_signal)

    conv_op_y1 = tf.split(y1, num_filters, axis=0)
    conv_op_y1 = K.concatenate(conv_op_y1, axis=1)
    conv_op_y1 = K.dot(conv_op_y1, ar_kernel)

    conv_op_y2 = tf.split(y2, num_filters, axis=0)
    conv_op_y2 = K.concatenate(conv_op_y2, axis=1)
    conv_op_y2 = K.dot(conv_op_y2, ma_kernel)

    conv_out = conv_op_y1 + conv_op_y2

    return conv_out
    # pylint: enable=no-value-for-parameter,redundant-keyword-arg,
    # pylint: enable=no-value-for-parameter,unexpected-keyword-arg

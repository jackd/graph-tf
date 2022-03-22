import tensorflow as tf


def get_type_spec(tensors):
    return tf.nest.map_structure(tf.type_spec_from_value, tensors)

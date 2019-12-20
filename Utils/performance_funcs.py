import tensorflow as tf


def calculate_inception_score(pyx: tf.Tensor, py: tf.Tensor) -> tf.Tensor:
    ς = 1.e-20
    py = broadcast_to_sample_size(py, sample_size=pyx.shape[0])
    # noinspection PyTypeChecker
    ops_matrix = pyx * (tf.math.log(pyx + ς) - tf.math.log(py + ς))
    inception_score = tf.math.exp(tf.math.reduce_mean(tf.math.reduce_sum(ops_matrix, axis=1), axis=0))
    return inception_score


def broadcast_to_sample_size(tensor: tf.Tensor, sample_size: int):
    reshaped_tensor = tf.reshape(tensor, shape=(1,) + tensor.numpy().shape)
    broadcast_tensor = tf.broadcast_to(reshaped_tensor, shape=(sample_size,) + tensor.numpy().shape)
    return broadcast_tensor

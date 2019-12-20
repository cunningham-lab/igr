import unittest
import numpy as np
import tensorflow as tf
from Models.VAENet import PlanarFlowLayer


class TestVAENet(unittest.TestCase):

    def test_planar_flow_computation(self):
        test_tolerance = 1.e-1
        sample_size = 1
        batch_n = 3
        example = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]]).T
        u = np.array([[-1., 1.1, 2.3, -2.], [4., 0.1, 0., -1.]]).T
        w = np.array([[1., 0., 3., 0.], [0., 3., 0., 1.]]).T
        b = np.array([1., -1])
        latent_n, var_num = example.shape
        example = np.reshape(example, newshape=(1, latent_n, 1, var_num))
        example = np.broadcast_to(example, shape=(batch_n, latent_n, sample_size, var_num))
        u2 = np.reshape(u, newshape=(1, latent_n, 1, var_num))
        w2 = np.reshape(w, newshape=(1, latent_n, 1, var_num))
        b2 = np.reshape(b, newshape=(1, 1, 1, var_num))
        u_tf = tf.constant(u2, dtype=tf.float32)
        b_tf = tf.constant(b2, dtype=tf.float32)
        w_tf = tf.constant(w2, dtype=tf.float32)

        pf = PlanarFlowLayer(units=latent_n)
        pf.build(input_shape=example.shape)
        pf.w = w_tf
        pf.u = u_tf
        pf.b = b_tf
        result_tf = pf.call(tf.constant(example, dtype=tf.float32)).numpy()
        # result_tf = pf(tf.constant(example, dtype=tf.float32)).numpy()
        result_np = compute_pf(example, w=w, u=u, b=b)

        diff = np.linalg.norm(result_tf - result_np) / np.linalg.norm(result_np)
        print(diff)
        self.assertTrue(expr=diff < test_tolerance)

    def test_planar_flow_layer_gradient(self):
        batch_n = 5
        latent_n = 10 - 1
        sample_size = 1
        var_num = 2
        shape = (batch_n, latent_n, sample_size, var_num)
        eps = tf.random.normal(shape=shape)
        pf = PlanarFlowLayer(units=latent_n)
        pf.build(input_shape=shape)

        with tf.GradientTape() as tape:
            output = pf.call(inputs=eps)

        gradient = tape.gradient(target=output, sources=pf.trainable_variables)
        self.assertTrue(gradient is not None)

    def test_planar_flow_layer_concat_gradient(self):
        batch_n = 5
        latent_n = 10 - 1
        sample_size = 1
        var_num = 2
        shape = (batch_n, latent_n, sample_size, var_num)
        eps = tf.random.normal(shape=shape)
        pf = tf.keras.Sequential([
            PlanarFlowLayer(units=latent_n),
            PlanarFlowLayer(units=latent_n),
            PlanarFlowLayer(units=latent_n)])

        with tf.GradientTape() as tape:
            output = pf(eps)

        gradient = tape.gradient(target=output, sources=pf.trainable_variables)
        self.assertTrue(len(gradient) == 3 * 3)


def compute_pf(inputs, w, u, b):
    batch_n, latent_n, sample_size, var_num = inputs.shape
    output = np.zeros(shape=inputs.shape)
    for batch in range(batch_n):
        for sample in range(sample_size):
            for var in range(var_num):
                output[batch, :, sample, var] = pf(z=inputs[batch, :, sample, var], w=w[:, var],
                                                   u=u[:, var], b=b[var])
    return output


def pf(z, w, u, b):
    alpha = -1 + np.log(1 + np.exp(np.dot(w, u))) - np.dot(w, u)
    u_tilde = u + alpha * w / np.linalg.norm(w)
    tanh = np.tanh(np.dot(w, z) + b)
    output = z + u_tilde * tanh
    return output


if __name__ == '__main__':
    unittest.main()

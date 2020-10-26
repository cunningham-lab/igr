import unittest
import numpy as np
import tensorflow as tf
from Models.VAENet import PlanarFlowLayer
from Models.VAENet import create_nested_planar_flow
from Models.VAENet import offload_weights_planar_flow
from Models.VAENet import generate_random_planar_flow_weights


class TestVAENet(unittest.TestCase):

    def test_planar_flow_weight_offloading(self):
        tolerance = 1.e-10
        nested_layers, latent_n, var_num = 2, 4, 1
        weights = generate_random_planar_flow_weights(nested_layers, latent_n, var_num)
        pf = offload_weights_planar_flow(weights)
        print(f'\nTEST: Weigth Offloading')
        for idx in range(len(pf.weights)):
            approx = weights[idx].numpy()
            ans = pf.weights[idx].numpy()
            diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans + 1.e-20)
            print(f'\nDiff {diff:1.3e}')
            self.assertTrue(expr=diff < tolerance)

    def test_nested_planar_flow_creation(self):
        tolerance = 1.e-1
        batch_n = 1
        sample_size = 1
        nested_layers = 2
        initializer = 'zeros'
        example = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]]).T
        latent_n, var_num = example.shape
        example = np.reshape(example, newshape=(1, latent_n, 1, var_num))
        example = np.broadcast_to(example, shape=(batch_n, latent_n, sample_size, var_num))
        planar_flow = create_nested_planar_flow(nested_layers, latent_n, var_num, initializer)
        approx = planar_flow(tf.constant(example, dtype=tf.float32)).numpy()
        diff = np.linalg.norm(approx - example) / np.linalg.norm(example)
        print(f'\nTEST: Nested Creation')
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_planar_flow_determinant(self):
        tolerance = 1.e-10
        categories_n = 4
        z = np.array([1 / np.sqrt(categories_n) for _ in np.arange(categories_n)])
        w = np.array([[-1 for _ in np.arange(categories_n)],
                      [1 / np.sqrt(categories_n) for i in range(categories_n, 0, -1)]]).T
        u = np.array([[-1 for _ in np.arange(categories_n)],
                      [i for i in range(categories_n, 0, -1)]]).T
        b = np.array([np.sum(z), -1.])
        nested_layers = len(b)
        approx = calculate_pf_log_det_np(z, w, u, b)
        u_tilde = np.zeros(shape=(categories_n, len(b)))
        for l in range(nested_layers):
            u_tilde[:, l] = get_u_tilde(u[:, l], w[:, l])

        ans = np.stack((1 + u_tilde[:, 0].T @ w[:, 0],
                        1 + u_tilde[:, 1] @ w[:, 1])).T
        ans = -np.sum(np.log(ans))
        diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
        print(f'\nTEST: Planar Flow Determinant')
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_planar_flow_computation(self):
        tolerance = 1.e-10
        categories_n = 4
        z = np.array([[i + 1 for i in np.arange(categories_n)],
                      [0 for _ in range(categories_n)],
                      [1 / np.sqrt(categories_n) for i in range(categories_n, 0, -1)]]).T
        w = np.array([[-1 for _ in np.arange(categories_n)],
                      [2 * i for i in range(categories_n)],
                      [1 / np.sqrt(categories_n) for i in range(categories_n, 0, -1)]]).T
        u = np.array([[-1 for _ in np.arange(categories_n)],
                      [i for i in range(categories_n)],
                      [i for i in range(categories_n, 0, -1)]]).T
        b = np.array([np.sum(z[:, 0]), 1., 0.])
        u_tilde = np.zeros(shape=(categories_n, len(b)))
        approx = np.zeros(shape=(categories_n, len(b)))
        for idx in range(len(b)):
            approx[:, idx] = planar_flow(z[:, idx], w[:, idx], u[:, idx], b[idx])
            u_tilde[:, idx] = get_u_tilde(u[:, idx], w[:, idx])
        ans = np.stack((z[:, 0],
                        u_tilde[:, 1] * np.tanh(1.),
                        z[:, 2] + u_tilde[:, 2] * np.tanh(1.))).T
        diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
        print(f'\nTEST: NumPy Planar Flow Manual Computation')
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_planar_flow_broadcast_computation(self):
        test_tolerance = 1.e-1
        sample_size = 1
        batch_n = 3
        z = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]]).T
        u = np.array([[-1., 1.1, 2.3, -2.], [4., 0.1, 0., -1.]]).T
        w = np.array([[1., 0., 3., 0.], [0., 3., 0., 1.]]).T
        b = np.array([1., -1])
        latent_n, var_num = z.shape
        z, u_tf, b_tf, w_tf = prepare_case(z, u, w, b, batch_n, sample_size)

        pf = PlanarFlowLayer(units=latent_n, var_num=var_num)
        pf.build(input_shape=z.shape)
        pf.w, pf.u, pf.b = w_tf, u_tf, b_tf
        result_tf = pf.call(tf.constant(z, dtype=tf.float32)).numpy()
        result_np = compute_pf(z, w=w, u=u, b=b)

        diff = np.linalg.norm(result_tf - result_np) / np.linalg.norm(result_np)
        print(f'\nTEST: Planar Flow Implementation')
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def test_planar_flow_layer_gradient(self):
        batch_n = 5
        latent_n = 10 - 1
        sample_size = 1
        var_num = 2
        shape = (batch_n, latent_n, sample_size, var_num)
        eps = tf.random.normal(shape=shape)
        pf = PlanarFlowLayer(units=latent_n, var_num=1)
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
            PlanarFlowLayer(units=latent_n, var_num=1),
            PlanarFlowLayer(units=latent_n, var_num=1),
            PlanarFlowLayer(units=latent_n, var_num=1)])

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
                output[batch, :, sample, var] = planar_flow(z=inputs[batch, :, sample, var],
                                                            w=w[:, var], u=u[:, var], b=b[var])
    return output


def planar_flow(z, w, u, b):
    u_tilde = get_u_tilde(u, w)
    output = z + u_tilde * np.tanh(np.dot(w, z) + b)
    return output


def get_u_tilde(u, w):
    alpha = -1 + np.log(1 + np.exp(np.dot(w, u))) - np.dot(w, u)
    u_tilde = u + alpha * w / np.linalg.norm(w)
    return u_tilde


def calculate_pf_log_det_np_all(inputs, w, u, b):
    batch_n, latent_n, sample_size, var_num = inputs.shape
    output = np.zeros(shape=(batch_n, 1, sample_size, var_num))
    for batch in range(batch_n):
        for sample in range(sample_size):
            for var in range(var_num):
                z_c = inputs[batch, :, sample, var].numpy()
                w_c = w[batch, :, sample, var, :]
                u_c = u[batch, :, sample, var, :]
                b_c = b[batch, :, sample, var, :].T
                output[batch, 0, sample, var] = calculate_pf_log_det_np(z_c, w_c, u_c, b_c)
    return output


def calculate_pf_log_det_np(z, w, u, b):
    nested_layers = len(b)
    z_nest = nest_planar_flow(nested_layers, z, w, u, b)
    log_det = 0
    for l in range(nested_layers):
        u_tilde = get_u_tilde(u[:, l], w[:, l])
        h_prime = 1 - np.tanh(w[:, l].T @ z_nest[:, l] + b[l]) ** 2
        log_det -= np.log(np.abs(1 + h_prime * u_tilde.T @ w[:, l]))
    return log_det


def nest_planar_flow(nested_layers, z, w, u, b):
    z_nest = np.zeros(shape=(z.shape + (nested_layers + 1, )))
    z_nest[:, 0] = z
    for l in range(1, nested_layers + 1):
        z_nest[:, l] = planar_flow(z_nest[:, l - 1], w[:, l - 1], u[:, l - 1], b[l - 1])
    return z_nest


def prepare_case(z, u, w, b, batch_n, sample_size):
    latent_n, var_num = z.shape
    z = np.reshape(z, newshape=(1, latent_n, 1, var_num))
    z = np.broadcast_to(z, shape=(batch_n, latent_n, sample_size, var_num))
    u2 = np.reshape(u, newshape=(1, latent_n, 1, var_num))
    w2 = np.reshape(w, newshape=(1, latent_n, 1, var_num))
    b2 = np.reshape(b, newshape=(1, 1, 1, var_num))
    u_tf = tf.constant(u2, dtype=tf.float32)
    b_tf = tf.constant(b2, dtype=tf.float32)
    w_tf = tf.constant(w2, dtype=tf.float32)
    return z, u_tf, b_tf, w_tf


if __name__ == '__main__':
    unittest.main()

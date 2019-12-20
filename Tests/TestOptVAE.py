# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Imports
# ===========================================================================================================
import unittest
import numpy as np
import tensorflow as tf
from Models.OptVAE import calculate_kl_norm_via_analytical_formula, calculate_categorical_closed_kl
from Models.OptVAE import calculate_kl_norm_via_general_analytical_formula
# ===========================================================================================================


class TestSBDist(unittest.TestCase):

    def test_calculate_kl_norm_via_analytical_formula(self):
        test_tolerance = 1.e-5
        samples_n = 1
        num_of_vars = 2
        μ = broadcast_to_shape(np.array([[1., -1., 0.15, 0.45, -0.3],
                                         [-0.1, 0.98, 0.02, -1.4, 0.35],
                                         [0., 0., 2., 2., 0.]]), samples_n=samples_n, num_of_vars=num_of_vars)
        log_σ2 = broadcast_to_shape(np.array(
            [[-1.7261764, 0.1970883, -0.05951275, 0.43101027, 1.00751897],
             [-1.55425, -0.0337, 1.22609, -0.19088, 0.9577],
             [0., 0., 0., 0., 0.]]), samples_n=samples_n, num_of_vars=num_of_vars)
        kl_norm_ans = calculate_kl_norm(μ=μ, σ2=np.exp(log_σ2))
        kl_norm = calculate_kl_norm_via_analytical_formula(mean=tf.constant(μ, dtype=tf.float32),
                                                           log_var=tf.constant(log_σ2, dtype=tf.float32))
        relative_diff = np.linalg.norm((kl_norm.numpy() - kl_norm_ans) / kl_norm_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

        samples_n = 2
        batch_n = 68
        categories_n = 15
        μ = broadcast_to_shape(np.random.normal(size=(batch_n, categories_n)),
                               samples_n=samples_n, num_of_vars=num_of_vars)
        log_σ2 = broadcast_to_shape(np.random.lognormal(size=(batch_n, categories_n)),
                                    samples_n=samples_n, num_of_vars=num_of_vars)
        kl_norm_ans = calculate_kl_norm(μ=μ, σ2=np.exp(log_σ2))
        kl_norm = calculate_kl_norm_via_analytical_formula(mean=tf.constant(μ, dtype=tf.float32),
                                                           log_var=tf.constant(log_σ2, dtype=tf.float32))
        relative_diff = np.linalg.norm((kl_norm.numpy() - kl_norm_ans) / kl_norm_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_calculate_kl_gs_via_discrete_formula(self):
        test_tolerance = 1.e-5
        samples_n = 1
        num_of_vars = 1
        log_α = broadcast_to_shape(np.array([[1., -1., 0.15, 0.45, -0.3],
                                             [-0.1, 0.98, 0.02, -1.4, 0.35]]),
                                   samples_n=samples_n, num_of_vars=num_of_vars)[:, :, :, 0]
        kl_discrete_ans = calculate_kl_discrete(α=tf.math.softmax(log_α, axis=1))
        kl_discrete = calculate_categorical_closed_kl(log_α=tf.constant(log_α, dtype=tf.float32))
        relative_diff = np.linalg.norm((kl_discrete.numpy() - kl_discrete_ans) / kl_discrete_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_calculate_kl_norm_via_general_analytical_formula(self):
        test_tolerance = 1.e-15
        samples_n = 1
        num_of_vars = 2
        mean_0 = broadcast_to_shape(np.array([[1., -1., 0.15, 0.45, -0.3],
                                              [-0.1, 0.98, 0.02, -1.4, 0.35],
                                              [0., 0., 0., 0., 0.]]),
                                    samples_n=samples_n, num_of_vars=num_of_vars)
        log_var_0 = broadcast_to_shape(np.array([[-1.724, 0.197, -0.0595, 0.431, 1.07],
                                                 [-1.55425, -0.0337, 1.22609, -0.19088, 0.9577],
                                                 [0., 0., 0., 0., 0.]]),
                                       samples_n=samples_n, num_of_vars=num_of_vars)
        mean_1 = broadcast_to_shape(np.array([[-0.2, 0.6, 1.5, -0.28, 2.],
                                              [0., 0., 0., 0., 0.],
                                              [0., 0., 0., 0., 0.]]),
                                    samples_n=samples_n, num_of_vars=num_of_vars)
        log_var_1 = broadcast_to_shape(np.array([[0.3, -0.4, 1.3, 0.5, -0.3],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.]]),
                                       samples_n=samples_n, num_of_vars=num_of_vars)
        kl_norm_ans = calculate_kl_norm_general_from_vectors(mean_0=mean_0, log_var_0=log_var_0,
                                                             mean_1=mean_1, log_var_1=log_var_1)
        kl_norm = calculate_kl_norm_via_general_analytical_formula(mean_0=mean_0, log_var_0=log_var_0,
                                                                   mean_1=mean_1, log_var_1=log_var_1)
        relative_diff = np.linalg.norm((kl_norm.numpy() - kl_norm_ans))
        self.assertTrue(expr=relative_diff < test_tolerance)

        self.assertTrue(expr=np.isclose(kl_norm.numpy()[-1, 0, 0], 0))


def calculate_kl_norm(μ, σ2):
    categories_n = μ.shape[1]
    kl_norm = np.zeros(shape=μ.shape)
    for i in range(categories_n):
        kl_norm[:, i, :] = 0.5 * (σ2[:, i, :] + μ[:, i, :] ** 2 - np.log(σ2[:, i, :]) - 1)
    return np.sum(kl_norm, axis=1)


def calculate_kl_norm_general_from_vectors(mean_0, log_var_0, mean_1, log_var_1):
    batch_n, categories_n, samples_n, num_of_var = mean_0.shape
    kl_norm = np.zeros(shape=(batch_n, samples_n, num_of_var))
    for i in range(batch_n):
        for j in range(samples_n):
            for r in range(num_of_var):
                cov_0 = np.diag(np.exp(log_var_0[i, :, j, r]))
                cov_1 = np.diag(np.exp(log_var_1[i, :, j, r]))
                kl_norm[i, j, r] = calculate_kl_norm_general(mean_0=mean_0[i, :, j, r], cov_0=cov_0,
                                                             mean_1=mean_1[i, :, j, r], cov_1=cov_1)
    return kl_norm


def calculate_kl_norm_general(mean_0, cov_0, mean_1, cov_1):
    categories_n = mean_0.shape[0]
    trace_term = np.trace(np.linalg.inv(cov_1) @ cov_0)
    means_term = (mean_1 - mean_0).T @ np.linalg.inv(cov_1) @ (mean_1 - mean_0)
    log_det_term = np.log(np.linalg.det(cov_1) / np.linalg.det(cov_0))
    kl_norm = 0.5 * (trace_term + means_term + log_det_term - categories_n)
    return kl_norm


def calculate_kl_discrete(α):
    categories_n = α.shape[1]
    kl_discrete = np.zeros(shape=α.shape)
    for i in range(categories_n):
        kl_discrete[:, i, :] = α[:, i, :] * (np.log(α[:, i, :]) - np.log(1 / categories_n))
    return np.sum(kl_discrete, axis=1)


def broadcast_to_shape(v, samples_n, num_of_vars):
    original_shape = v.shape
    v = np.reshape(v, newshape=original_shape + (1, 1))
    v = np.broadcast_to(v, shape=original_shape + (samples_n, 1))
    v = np.broadcast_to(v, shape=original_shape + (samples_n, num_of_vars))
    return v


# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# If main block
# ===========================================================================================================
if __name__ == '__main__':
    unittest.main()
# ===========================================================================================================

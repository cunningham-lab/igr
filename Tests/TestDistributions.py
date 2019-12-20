# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Imports
# ===========================================================================================================
import unittest
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from scipy.special import logsumexp, loggamma
from Utils.Distributions import SBDist, compute_log_logit_normal, compute_log_jac, compute_log_sb_dist
from Utils.Distributions import iterative_sb_and_jac, generate_lower_and_upper_triangular_matrices_for_sb
from Utils.Distributions import project_to_vertices_via_random_jump
from Utils.Distributions import compute_log_exp_gs_dist
# ===========================================================================================================


class TestDistributions(unittest.TestCase):

    def test_compute_log_exp_gs_dist(self):
        test_tolerance = 1.e-5
        sample_size = 1
        categories_n = 10
        num_of_vars = 3
        batch_n = 2
        τ = tf.constant(value=0.6, dtype=tf.float32)
        log_π = tf.random.normal(shape=(batch_n, categories_n, sample_size, num_of_vars))

        uniform_sample = np.random.uniform(size=log_π.shape)
        gumbel_sample = tf.constant(-np.log(-np.log(uniform_sample)), dtype=tf.float32)
        y = (log_π + gumbel_sample) / τ
        log_ψ = y.numpy() - logsumexp(y.numpy(), keepdims=True)
        log_exp_gs_ans = calculate_log_exp_concrete_for_tensor(log_ψ=log_ψ, α=tf.math.exp(log_π), τ=τ)
        log_exp_gs = compute_log_exp_gs_dist(log_psi=log_ψ, logits=log_π, temp=τ)

        relative_diff = tf.linalg.norm(log_exp_gs.numpy() - log_exp_gs_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

        π = tf.constant(np.random.dirichlet(alpha=np.array([i + 1 for i in range(categories_n)]),
                                            size=(batch_n, sample_size, num_of_vars)), dtype=tf.float32)
        π = tf.transpose(π, perm=[0, 3, 1, 2])
        y = (tf.math.log(π) + gumbel_sample) / τ
        log_ψ = y.numpy() - logsumexp(y.numpy(), keepdims=True)
        log_exp_gs_ans = calculate_log_exp_concrete_for_tensor(log_ψ=log_ψ, α=π, τ=τ)
        log_exp_gs = compute_log_exp_gs_dist(log_psi=log_ψ, logits=tf.math.log(π), temp=τ)
        relative_diff = tf.linalg.norm(log_exp_gs.numpy() - log_exp_gs_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_perform_stick_break_with_manual_input(self):
        test_tolerance = 1.e-7
        sample_size = 3
        threshold = 0.8
        kappa_stick = np.array([[0.1, 0.22222224, 0.42857146, 0.75000006],
                                [0.3, 0.4, 0.5, 0.6],
                                [0.3, 0.28571428571428575, 0.2, 0.5]])
        batch_size, max_size = kappa_stick.shape[0], kappa_stick.shape[1]
        kappa_stick = broadcast_to_sample_size(kappa_stick, shape=kappa_stick.shape,
                                               sample_size=sample_size)
        eta_ans = np.array([[0.1, 0.2, 0.3, 0.3],
                            [0.3, 0.28, 0.21, 0.126],
                            [0.3, 0.2, 0.1, 0.2]])
        # eta_cumsum = np.array([[0.1, 0.3,  0.6,  0.9],
        #                        [0.3, 0.58, 0.79, 0.916],
        #                        [0.3, 0.5,  0.6,  0.8]])
        eta_ans = broadcast_to_sample_size(eta_ans, shape=eta_ans.shape, sample_size=sample_size)
        n_required_ans = 2 + 1
        eta_ans = eta_ans[:, :n_required_ans, :]
        mu = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1))
        xi = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1))
        sb_dist = SBDist(mu=mu, xi=xi, sample_size=1, threshold=threshold)
        sb_dist.lower, sb_dist.upper = generate_lower_and_upper_triangular_matrices_for_sb(
            categories_n=sb_dist.categories_n, lower=sb_dist.lower, upper=sb_dist.upper,
            batch_size=sb_dist.batch_size, sample_size=sb_dist.sample_size)
        sb_dist.truncation_option = 'quantile'
        eta, _ = sb_dist.perform_stick_break_and_compute_log_jac(kappa=kappa_stick)
        eta_iter, _ = sb_dist.compute_sb_and_log_jac_iteratively(κ=kappa_stick)

        eta_test = calculate_η_from_κ(kappa_stick)[:, :n_required_ans, :]
        relative_diff = tf.linalg.norm(eta_test - eta_ans) / tf.linalg.norm(eta_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

        relative_diff = tf.linalg.norm(eta - eta_ans) / tf.linalg.norm(eta_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

        relative_diff = tf.linalg.norm(eta_iter - eta_ans) / tf.linalg.norm(eta_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_perform_stick_break_with_generated_kappa(self):
        #    Global parameters  #######
        test_tolerance = 1.e-7
        batch_size, max_size, sample_size = 2, 20, 10
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        mu = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1))
        xi = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1))
        for threshold in thresholds:
            sb_dist = SBDist(mu=mu, xi=xi, sample_size=sample_size, threshold=threshold)
            sb_dist.lower, sb_dist.upper = generate_lower_and_upper_triangular_matrices_for_sb(
                categories_n=sb_dist.categories_n, lower=sb_dist.lower, upper=sb_dist.upper,
                batch_size=sb_dist.batch_size, sample_size=sb_dist.sample_size)
            kappa = np.array([1 / (2 ** (i + 1)) for i in range(max_size)])
            kappa = np.broadcast_to(kappa, shape=(batch_size, max_size))
            kappa = broadcast_to_sample_size(kappa, shape=(batch_size, max_size), sample_size=sample_size)
            eta_ans = calculate_η_from_κ(kappa)

            eta, _ = sb_dist.perform_stick_break_and_compute_log_jac(kappa=kappa)
            n_required = eta.shape[1]
            eta_ans = eta_ans[:, :n_required, :]
            relative_diff = np.linalg.norm(eta.numpy() - eta_ans) / np.linalg.norm(eta_ans)
            self.assertTrue(expr=relative_diff < test_tolerance)

            eta_it, _ = sb_dist.compute_sb_and_log_jac_iteratively(κ=kappa)
            relative_diff = np.linalg.norm(eta_it.numpy() - eta_ans) / np.linalg.norm(eta_ans)
            self.assertTrue(expr=relative_diff < test_tolerance)

    def test_compute_log_jac(self):
        test_tolerance = 1.e-7
        sample_size = 3
        #   Jacobian Test   #######
        kappa_stick = np.array([[0.1, 0.22222224, 0.42857146, 0.75000006],
                               [0.3, 0.4, 0.5, 0.6]])
        # eta = np.array([[0.1, 0.2,  0.3,  0.3],
        #                 [0.3, 0.28, 0.21, 0.126]])  # this is the corresponding eta
        # eta_cumsum = np.array([[0.1, 0.3,  0.6,  0.9],
        #                        [0.3, 0.58, 0.79, 0.916]])
        max_size, batch_size = kappa_stick.shape[1], kappa_stick.shape[0]
        kappa_stick = broadcast_to_sample_size(kappa_stick, sample_size=sample_size,
                                               shape=(batch_size, max_size))

        mu = broadcast_to_sample_size(np.array([1 for _ in range(max_size)]),
                                      shape=(1, max_size), sample_size=sample_size)
        xi = broadcast_to_sample_size(np.array([1 for _ in range(max_size)]),
                                      shape=(1, max_size), sample_size=sample_size)
        sb_dist = SBDist(mu=mu, xi=xi, sample_size=1, threshold=0.99)
        sb_dist.lower, sb_dist.upper = generate_lower_and_upper_triangular_matrices_for_sb(
            categories_n=sb_dist.categories_n, lower=sb_dist.lower, upper=sb_dist.upper,
            batch_size=sb_dist.batch_size, sample_size=sb_dist.sample_size)
        sb_dist.truncation_option = 'max'
        _, jac = sb_dist.perform_stick_break_and_compute_log_jac(kappa=kappa_stick)

        jac_ans = np.array([np.log(1 * (1 / (1 - 0.1)) * (1 / (1 - 0.3)) * (1 / (1 - 0.6))),
                            np.log(1 * (1 / (1 - 0.3)) * (1 / (1 - 0.58)) * (1 / (1 - 0.79)))])
        jac_ans = broadcast_to_sample_size(jac_ans, shape=(batch_size,), sample_size=sample_size)

        jac_test = calculate_log_jac_η_from_κ(κ=kappa_stick.numpy())

        jac_inverse = compute_log_jac(κ=kappa_stick)
        _, jac_iter = iterative_sb_and_jac(κ=kappa_stick)

        relative_diff = tf.linalg.norm(jac_test - jac_ans) / tf.linalg.norm(jac_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

        relative_diff = tf.linalg.norm(jac_iter - jac_ans) / tf.linalg.norm(jac_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

        relative_diff = tf.linalg.norm(jac_inverse - jac_ans) / tf.linalg.norm(jac_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

        relative_diff = tf.linalg.norm(jac - jac_ans) / tf.linalg.norm(jac_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_random_jump_projection_with_manual_case(self):
        test_tolerance = 1.e-7
        sample_size = 1
        uniform_threshold = tf.constant(0.5)
        uniform_sample = np.array([0.1, 0.7, 0.3])
        temp = tf.constant(0.1)
        eta = np.array([[0.8, 0.05, 0., 0.05],
                        [0.3, 0.2, 0.05, 0.35],
                        [0.1, 0.8, 0., 0.]])
        batch_size, categories_n = eta.shape
        eta = broadcast_to_sample_size(eta, shape=(batch_size, categories_n), sample_size=sample_size)
        psi_ans = tf.constant(np.array([[0.8, 0.05, 0., 0.05],
                                        [0.03, 0.02, 0.005, 0.935],
                                        [0.1, 0.8, 0., 0.]]), dtype=tf.float32)
        psi_ans = broadcast_to_sample_size(psi_ans, shape=(batch_size, categories_n),
                                           sample_size=sample_size)
        with tf.GradientTape() as tape:
            tape.watch(eta)
            lam, psi = project_to_vertices_via_random_jump(eta=eta, temp=temp, uniform_sample=uniform_sample,
                                                           random_jump_threshold=uniform_threshold)
        grad = tape.gradient(target=psi, sources=eta)
        relative_diff = tf.linalg.norm(psi - psi_ans) / tf.linalg.norm(psi_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)
        self.assertTrue(expr=grad is not None)

    # def test_log_sb_dist_softmax(self):
    #     #    Global parameters  #######
    #     test_tolerance = 1.e-5
    #     temp = tf.constant(value=0.1, dtype=tf.float32)
    #     ς = 1.e-20
    #     batch_size, max_size, sample_size = 3, 25, 100
    #
    #     # Set up scenario
    #     mu = np.reshape(np.array([-4 + i for i in range(max_size)]), newshape=(1, max_size, 1))
    #     xi = np.reshape(np.array([0.55 for _ in range(max_size)]), newshape=(1, max_size, 1))
    #     mu = tf.constant(np.broadcast_to(mu, shape=(batch_size, max_size, 1)), dtype=tf.float32)
    #     xi = tf.constant(np.broadcast_to(xi, shape=(batch_size, max_size, 1)), dtype=tf.float32)
    #     sb = SBDist(mu=mu, xi=xi, temp=temp, sample_size=sample_size, threshold=0.99)
    #     sb.projection_option = 'softmax'
    #     sb.do_reparameterization_trick()
    #     log_q_η = compute_log_logit_normal(epsilon=sb.epsilon, sigma=sb.sigma, kappa=sb.kappa) + sb.log_jac
    #     log_q_λ = compute_log_sb_dist(lam=sb.lam, kappa=sb.kappa, sigma=sb.sigma, epsilon=sb.epsilon,
    #                                   log_jac=sb.log_jac, temp=sb.temp)
    #
    #     with tf.GradientTape() as tape:
    #         tape.watch(tensor=sb.eta)
    #         y = tf.math.log(sb.eta) / temp
    #     jac = tape.jacobian(target=y, sources=sb.eta)[:, :sb.psi.shape[1], :]
    #
    #     for batch_id in range(batch_size):
    #         log_jac = np.zeros(shape=sample_size)
    #         for i in range(sample_size):
    #             a = jac[batch_id, :, i, batch_id, :, i]
    #             log_jac[i] = -np.log(np.abs(np.linalg.det(a)) + ς)
    #
    #         log_q_ans = log_q_η[batch_id, :] + log_jac
    #         relative_diff = np.linalg.norm((log_q_ans - log_q_λ[batch_id, :]) / log_q_ans)
    #         self.assertTrue(expr=relative_diff < test_tolerance)

    # def test_compute_log_logit_normal(self):
    #     #    Global parameters  #######
    #     test_tolerance = 1.e-4
    #     temp = tf.constant(value=0.1, dtype=tf.float32)
    #     threshold = 0.99
    #
    #     batch_size, max_size, sample_size = 3, 25, 2
    #
    #     mu = np.random.normal(loc=0, scale=0.01, size=(batch_size, max_size, 1))
    #     xi = np.random.normal(loc=1, scale=0.01, size=(batch_size, max_size, 1))
    #     mu = tf.constant(value=mu, dtype=tf.float32)
    #     xi = tf.constant(value=xi, dtype=tf.float32)
    #
    #     sb_dist = SBDist(mu=mu, xi=xi, temp=temp, sample_size=sample_size, threshold=threshold)
    #     sb_dist.projection_option = 'softmax'
    #     sb_dist.do_reparameterization_trick()
    #
    #     log_q_kappa = compute_log_logit_normal(epsilon=sb_dist.epsilon, sigma=sb_dist.sigma,
    #                                            kappa=sb_dist.kappa)
    #
    #     # Compute the function via with sciPy
    #     for batch_id in range(batch_size):
    #         mu = tf.broadcast_to(input=sb_dist.mu[batch_id, :sb_dist.n_required, :],
    #                              shape=(sb_dist.n_required, sb_dist.sample_size))
    #         kappa = sb_dist.kappa[batch_id, :, :]
    #         delta = sb_dist.delta[batch_id, :, :]
    #         sigma = sb_dist.sigma[batch_id, :, :]
    #
    #         kappa_things = -tf.reduce_sum(tf.math.log(kappa * (1 - kappa)), axis=0)
    #         normal = norm.pdf(delta.numpy(), loc=mu.numpy(), scale=sigma.numpy())
    #         log_norm = np.sum(np.log(normal), axis=0)
    #         log_norm_kappa = log_norm + kappa_things
    #
    #         relative_diff = np.linalg.norm((log_q_kappa[batch_id, :] - log_norm_kappa) / log_norm_kappa)
    #         self.assertTrue(expr=relative_diff < test_tolerance)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Test Functions
# ===========================================================================================================
def calculate_log_exp_concrete_for_tensor(log_ψ, α, τ):
    batch_n, sample_size, num_of_vars = log_ψ.shape[0], log_ψ.shape[2], log_ψ.shape[3]
    log_exp_concrete = np.zeros(shape=(batch_n, sample_size, num_of_vars))
    for b in range(batch_n):
        for s in range(sample_size):
            for v in range(num_of_vars):
                log_exp_concrete[b, s, v] = calculate_log_exp_concrete(log_ψ=log_ψ[b, :, s, v],
                                                                       α=α[b, :, s, v],
                                                                       τ=τ)
    return log_exp_concrete


def calculate_log_exp_concrete(log_ψ, α, τ):
    categories_n = log_ψ.shape[0]
    log_const = loggamma(categories_n) + (categories_n - 1) * np.log(τ)
    aux = np.log(α) - τ * log_ψ
    log_sum = np.sum(aux) - categories_n * logsumexp(aux)
    return log_const + log_sum


def calculate_log_jac_η_from_κ(κ):
    η = calculate_η_from_κ(κ=κ)
    log_jac_η = calculate_log_jac_η(η=η)
    return log_jac_η


def calculate_log_jac_η(η):
    batch_size, n_required, sample_size = η.shape
    ς = 1.e-20
    log_jac_η = np.zeros(shape=(batch_size, n_required, sample_size))
    for batch_id in range(batch_size):
        for sample_id in range(sample_size):
            cumsum = 0
            for r in range(n_required):
                log_jac_η[batch_id, r, sample_id] += -np.log(1. - cumsum + ς)
                cumsum += η[batch_id, r, sample_id]
    return np.sum(log_jac_η, axis=1)


def calculate_η_from_κ(κ):
    batch_size, n_required, sample_size = κ.shape
    η = np.zeros(shape=(batch_size, n_required, sample_size))
    for batch_id in range(batch_size):
        for sample_id in range(sample_size):
            cumsum = 0
            for r in range(n_required):
                η[batch_id, r, sample_id] = κ[batch_id, r, sample_id] * (1. - cumsum)
                cumsum += η[batch_id, r, sample_id]
    return η


def broadcast_to_batch_and_sample_size(a, batch_n, sample_size):
    shape = a.shape
    a = np.reshape(a, newshape=(1,) + shape)
    a = np.broadcast_to(a, shape=(batch_n,) + shape)
    a = broadcast_to_sample_size(a=a, shape=a.shape, sample_size=sample_size)
    return a


def broadcast_to_sample_size(a, shape, sample_size):
    a = np.reshape(a, newshape=shape + (1, ))
    a = np.broadcast_to(a, shape=shape + (sample_size, ))
    a = tf.constant(value=a, dtype=tf.float32)
    return a
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# If main block
# ===========================================================================================================


if __name__ == '__main__':
    unittest.main()
# ===========================================================================================================

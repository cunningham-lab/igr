# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Imports
# ===========================================================================================================
import unittest
import numpy as np
import tensorflow as tf
from scipy.special import logsumexp, loggamma
from Utils.Distributions import IGR_SB, IGR_SB_Finite
from Utils.Distributions import compute_log_exp_gs_dist, project_to_vertices_via_softmax_pp
# ===========================================================================================================


class TestDistributions(unittest.TestCase):

    def test_softmaxpp(self):
        test_tolerance = 1.e-4
        batch_size, categories_n, sample_size, num_of_vars = 2, 3, 4, 5
        lam = tf.constant(0., shape=(batch_size, categories_n - 1, sample_size, num_of_vars))
        psi_ans = compute_softmaxpp_for_all(lam=lam.numpy(), delta=0.1)
        psi = project_to_vertices_via_softmax_pp(lam).numpy()
        relative_diff = np.linalg.norm(psi - psi_ans) / np.linalg.norm(psi_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_compute_log_exp_gs_dist(self):
        test_tolerance = 1.e-4
        batch_n, categories_n, sample_size, num_of_vars = 2, 10, 1, 3
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

    def test_perform_finite_stick_break_with_manual_input(self):
        test_tolerance = 1.e-7
        sample_size, num_of_vars = 5, 2
        temp = tf.constant(0.1, dtype=tf.float32)
        kappa_stick = np.array([[0.1, 0.22222224, 0.42857146, 0.75000006],
                                [0.3, 0.4, 0.5, 0.6],
                                [0.3, 0.28571428571428575, 0.2, 0.5]])
        batch_size, max_size = kappa_stick.shape[0], kappa_stick.shape[1]
        kappa_stick = broadcast_sample_and_num(kappa_stick, kappa_stick.shape, sample_size, num_of_vars)
        eta_ans = np.array([[0.1, 0.2, 0.3, 0.3],
                            [0.3, 0.28, 0.21, 0.126],
                            [0.3, 0.2, 0.1, 0.2]])
        eta_ans = broadcast_sample_and_num(eta_ans, eta_ans.shape, sample_size, num_of_vars)
        mu = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1, num_of_vars))
        xi = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1, num_of_vars))

        sb = IGR_SB_Finite(mu, xi, temp, sample_size=sample_size)
        eta_matrix = compute_and_threshold_eta(sb, kappa_stick, run_iteratively=False)
        eta_iter = compute_and_threshold_eta(sb, kappa_stick, run_iteratively=True)
        eta_np = calculate_eta_from_kappa(kappa_stick)

        eta_all = [eta_np, eta_matrix.numpy(), eta_iter.numpy()]
        for e in eta_all:
            relative_diff = np.linalg.norm(e - eta_ans) / np.linalg.norm(eta_ans)
            self.assertTrue(expr=relative_diff < test_tolerance)

    def test_perform_stick_break_with_manual_input(self):
        test_tolerance = 1.e-7
        sample_size, num_of_vars, threshold = 5, 2, 0.8
        temp = tf.constant(0.1, dtype=tf.float32)
        kappa_stick = np.array([[0.1, 0.22222224, 0.42857146, 0.75000006],
                                [0.3, 0.4, 0.5, 0.6],
                                [0.3, 0.28571428571428575, 0.2, 0.5]])
        batch_size, max_size = kappa_stick.shape[0], kappa_stick.shape[1]
        kappa_stick = broadcast_sample_and_num(kappa_stick, kappa_stick.shape, sample_size, num_of_vars)
        eta_ans = np.array([[0.1, 0.2, 0.3, 0.3],
                            [0.3, 0.28, 0.21, 0.126],
                            [0.3, 0.2, 0.1, 0.2]])
        # eta_cumsum = np.array([[0.1, 0.3,  0.6,  0.9],
        #                        [0.3, 0.58, 0.79, 0.916],
        #                        [0.3, 0.5,  0.6,  0.8]])
        n_required_ans = 2 + 1
        eta_ans = broadcast_sample_and_num(eta_ans, eta_ans.shape, sample_size, num_of_vars)
        eta_ans = eta_ans[:, :n_required_ans, :]
        mu = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1, num_of_vars))
        xi = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1, num_of_vars))

        sb = IGR_SB(mu, xi, temp, sample_size=sample_size, threshold=threshold)
        eta_matrix = compute_and_threshold_eta(sb, kappa_stick, run_iteratively=False)
        eta_iter = compute_and_threshold_eta(sb, kappa_stick, run_iteratively=True)
        eta_np = calculate_eta_from_kappa(kappa_stick)[:, :n_required_ans, :]

        eta_all = [eta_np, eta_matrix.numpy(), eta_iter.numpy()]
        for e in eta_all:
            relative_diff = np.linalg.norm(e - eta_ans) / np.linalg.norm(eta_ans)
            self.assertTrue(expr=relative_diff < test_tolerance)

    def test_perform_stick_break_with_generated_kappa(self):
        #    Global parameters  #######
        test_tolerance = 1.e-7
        temp = tf.constant(0.1, dtype=tf.float32)
        batch_size, max_size, sample_size, num_of_vars = 2, 20, 10, 4
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        mu = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1, num_of_vars))
        xi = tf.constant(value=1, dtype=tf.float32, shape=(batch_size, max_size, 1, num_of_vars))
        for threshold in thresholds:
            sb_dist = IGR_SB(mu=mu, xi=xi, temp=temp, sample_size=sample_size, threshold=threshold)
            kappa = np.array([1 / (2 ** (i + 1)) for i in range(max_size)])
            kappa = np.broadcast_to(kappa, shape=(batch_size, max_size))
            kappa = broadcast_sample_and_num(kappa, shape=(batch_size, max_size), sample_size=sample_size,
                                             num_of_vars=num_of_vars)
            eta_ans = calculate_eta_from_kappa(kappa)

            eta_matrix = compute_and_threshold_eta(sb_dist, kappa, run_iteratively=False)
            eta_iter = compute_and_threshold_eta(sb_dist, kappa, run_iteratively=True)
            n_required = eta_matrix.shape[1]
            eta_ans = eta_ans[:, :n_required, :, :]
            eta_all = [eta_matrix, eta_iter]
            for e in eta_all:
                relative_diff = np.linalg.norm(e.numpy() - eta_ans) / np.linalg.norm(eta_ans)
                self.assertTrue(expr=relative_diff < test_tolerance)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Test Functions
# ===========================================================================================================
def compute_softmaxpp_for_all(lam, delta=1.):
    batch_n, categories_n, sample_size, num_of_vars = lam.shape
    psi = np.zeros(shape=(batch_n, categories_n + 1, sample_size, num_of_vars))
    for b in range(batch_n):
        for s in range(sample_size):
            for v in range(num_of_vars):
                psi[b, :, s, v] = compute_softmaxpp(lam[b, :, s, v], delta)
    return psi


def compute_softmaxpp(lam, delta=1.):
    categories_n = lam.shape[0]
    psi = np.zeros(shape=categories_n + 1)
    exp_lam = np.exp(lam)
    sum_exp_lam = np.sum(exp_lam)
    psi[:categories_n] = exp_lam / (sum_exp_lam + delta)
    psi[-1] = 1 - np.sum(psi)
    return psi


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


def compute_and_threshold_eta(sb, kappa_stick, run_iteratively):
    sb.run_iteratively = run_iteratively
    eta = sb.apply_stick_break(kappa_stick)
    return eta


def calculate_eta_from_kappa(kappa):
    batch_size, n_required, sample_size, num_of_vars = kappa.shape
    eta = np.zeros(shape=(batch_size, n_required, sample_size, num_of_vars))
    for b in range(batch_size):
        for s in range(sample_size):
            for v in range(num_of_vars):
                cumsum = 0
                for r in range(n_required):
                    eta[b, r, s, v] = kappa[b, r, s, v] * (1. - cumsum)
                    cumsum += eta[b, r, s, v]
    return eta


def broadcast_to_batch_and_sample_size(a, batch_n, sample_size):
    shape = a.shape
    a = np.reshape(a, newshape=(1,) + shape)
    a = np.broadcast_to(a, shape=(batch_n,) + shape)
    a = broadcast_sample_and_num(a=a, shape=a.shape, sample_size=sample_size)
    return a


def broadcast_sample_and_num(a, shape, sample_size, num_of_vars):
    a = np.reshape(a, newshape=shape + (1, 1))
    a = np.broadcast_to(a, shape=shape + (sample_size, 1))
    a = np.broadcast_to(a, shape=shape + (sample_size, num_of_vars))
    a = tf.constant(value=a, dtype=tf.float32)
    return a


if __name__ == '__main__':
    unittest.main()

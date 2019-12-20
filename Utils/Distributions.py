# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
import numpy as np
import tensorflow as tf
from functools import partial
from typing import Tuple, List
from os import environ as os_env
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================


class Distributions:

    def __init__(self, batch_size: int, categories_n: int, sample_size: int = 1, num_of_vars: int = 1,
                 noise_type: str = 'normal',
                 temp: tf.Tensor = tf.constant(0.1, dtype=tf.float32), threshold: float = 0.99):

        self.noise_type = noise_type
        self.threshold = threshold
        self.temp = temp
        self.batch_size = batch_size
        self.categories_n = categories_n
        self.sample_size = sample_size
        self.num_of_vars = num_of_vars

        self.epsilon = tf.constant(0., dtype=tf.float32)
        self.sigma = tf.constant(0., dtype=tf.float32)
        self.delta = tf.constant(0., dtype=tf.float32)
        self.kappa = tf.constant(0., dtype=tf.float32)
        self.n_required = categories_n
        self.psi = tf.constant(0., dtype=tf.float32)

        self.truncation_option = 'quantile'
        self.quantile = 70
        self.log_q_psi = tf.constant(0., dtype=tf.float32)

    def broadcast_params_to_sample_size(self, params: list):
        params_broad = []
        for param in params:
            shape = (self.batch_size, self.categories_n, self.sample_size, self.num_of_vars)
            param_w_samples = tf.broadcast_to(input=param, shape=shape)
            params_broad.append(param_w_samples)
        return params_broad

    def sample_noise(self, shape) -> tf.Tensor:
        if self.noise_type == 'normal':
            epsilon = tf.random.normal(shape=shape)
        elif self.noise_type == 'trunc_normal':
            epsilon = tf.random.truncated_normal(shape=shape)
        elif self.noise_type == 'gamma':
            epsilon = tf.random.gamma(shape=shape, alpha=1., beta=1.)
        elif self.noise_type == 'cauchy':
            epsilon = tf.constant(np.random.standard_cauchy(size=shape), dtype=tf.float32)
        else:
            raise RuntimeError
        return epsilon

    def perform_truncation_via_threshold(self, vector):
        vector_cumsum = tf.math.cumsum(x=vector, axis=1)
        larger_than_threshold = tf.where(condition=vector_cumsum <= self.threshold)
        if self.truncation_option == 'quantile':
            self.n_required = int((np.percentile(larger_than_threshold[:, 1] + 1, q=self.quantile)))
        elif self.truncation_option == 'max':
            self.n_required = (tf.math.reduce_max(larger_than_threshold[:, 1]) + 1).numpy()
        else:
            self.n_required = (tf.math.reduce_mean(larger_than_threshold[:, 1]) + 1).numpy()

    def subset_variables_to_n_required(self, epsilon, sigma, delta, kappa):
        self.epsilon = epsilon[:, :self.n_required, :]
        self.sigma = sigma[:, :self.n_required, :]
        self.delta = delta[:, :self.n_required, :]
        self.kappa = kappa[:, :self.n_required, :]


class GaussianSoftmaxDist(Distributions):
    def __init__(self, mu: tf.Tensor, xi: tf.Tensor, noise_type: str = 'normal', sample_size: int = 1,
                 temp: tf.Tensor = tf.constant(0.1, dtype=tf.float32)):
        super().__init__(batch_size=mu.shape[0], categories_n=mu.shape[1], sample_size=sample_size,
                         noise_type=noise_type, temp=temp, num_of_vars=mu.shape[3])

        self.mu = mu
        self.xi = xi
        self.lam = tf.constant(0., dtype=tf.float32)
        self.psi = tf.constant(0., dtype=tf.float32)
        self.log_psi = tf.constant(0., dtype=tf.float32)

    def do_reparameterization_trick(self):
        mu_broad, xi_broad = self.broadcast_params_to_sample_size(params=[self.mu, self.xi])
        epsilon = self.sample_noise(shape=mu_broad.shape)
        sigma = convert_ξ_to_σ(ξ=xi_broad,)
        self.lam = (mu_broad + sigma * epsilon) / self.temp
        self.log_psi = self.lam - tf.math.reduce_logsumexp(self.lam, axis=1, keepdims=True)
        self.psi = self.project_to_vertices()

    def project_to_vertices(self):
        psi = project_to_vertices_via_softmax_pp(self.lam)
        # psi = tf.math.softmax(self.lam, axis=1)
        return psi


class IsoGauSoftMax(Distributions):
    def __init__(self, mu: tf.Tensor, noise_type: str = 'normal', temp: tf.Tensor = tf.constant(0.1),
                 sample_size: int = 1):
        super().__init__(batch_size=mu.shape[0], categories_n=mu.shape[1], noise_type=noise_type, temp=temp,
                         sample_size=sample_size, num_of_vars=mu.shape[3])

        self.mu = mu
        self.lam = tf.constant(0., dtype=tf.float32)
        self.psi = tf.constant(0., dtype=tf.float32)

    def do_reparameterization_trick(self):
        mu_broad = self.broadcast_params_to_sample_size(params=[self.mu])[0]
        epsilon = self.sample_noise(shape=mu_broad.shape)
        self.lam = (mu_broad + epsilon) / self.temp
        self.psi = self.project_to_vertices()

    def project_to_vertices(self):
        psi = tf.math.softmax(self.lam, axis=1)
        return psi


class GaussianSoftPlus(GaussianSoftmaxDist):
    def __init__(self, mu: tf.Tensor, xi: tf.Tensor, temp: tf.Tensor, sample_size: int = 1,
                 noise_type: str = 'normal'):
        super().__init__(mu=mu, xi=xi, noise_type=noise_type, temp=temp, sample_size=sample_size)

    def project_to_vertices(self):
        psi = project_to_vertices_via_softplus(lam=self.lam)
        return psi


class CauchySoftmaxDist(GaussianSoftmaxDist):
    def __init__(self, mu: tf.Tensor, xi: tf.Tensor, noise_type: str = 'cauchy', sample_size: int = 1,
                 temp: tf.Tensor = tf.constant(0.1, dtype=tf.float32)):
        super().__init__(mu=mu, xi=xi, noise_type=noise_type, temp=temp, sample_size=sample_size)
        self.noise_type = 'cauchy'


class LogitDist(Distributions):

    def __init__(self, mu: tf.Tensor, xi: tf.Tensor, sample_size: int = 1, noise_type: str = 'normal',
                 temp: tf.Tensor = tf.constant(0.1, dtype=tf.float32), threshold: float = 0.99):
        super().__init__(batch_size=mu.shape[0], categories_n=mu.shape[1], sample_size=sample_size,
                         noise_type=noise_type, temp=temp, threshold=threshold)

        self.mu = mu
        self.xi = xi
        self.eta = tf.constant(0., dtype=tf.float32)
        self.lam = tf.constant(0., dtype=tf.float32)
        self.projection_option = 'softmax'
        self.random_jump_threshold = 0.25

    def compute_log_logit_dist(self) -> tf.Tensor:
        if self.projection_option == 'softmax':
            log_q_psi = compute_log_logit_dist(epsilon=self.epsilon, sigma=self.sigma, kappa=self.kappa,
                                               temp=self.temp, lam=self.lam)
        else:
            log_q_psi = compute_log_logit_dist_projection(epsilon=self.epsilon, sigma=self.sigma,
                                                          kappa=self.kappa, temp=self.temp)
        return log_q_psi

    def do_reparameterization_trick(self):
        # mu_broad, xi_broad = self.broadcast_params_to_sample_size(params=[self.mu, self.xi])
        # mu_broad = self.mu
        # xi_broad = self.xi
        mu_broad = self.mu[:, :, :, 0]
        xi_broad = self.xi[:, :, :, 0]
        mu_broad = tf.broadcast_to(mu_broad, shape=(self.batch_size, self.categories_n, self.sample_size))
        xi_broad = tf.broadcast_to(xi_broad, shape=(self.batch_size, self.categories_n, self.sample_size))
        epsilon = self.sample_noise(shape=mu_broad.shape)
        sigma, delta, kappa = retrieve_transformations_up_to_kappa(mu_broad=mu_broad, xi_broad=xi_broad,
                                                                   epsilon=epsilon)
        self.get_eta_and_n_required(kappa=kappa)
        self.subset_variables_to_n_required(epsilon, sigma, delta, kappa)
        if self.projection_option == 'softmax':
            # self.lam, self.psi = project_to_vertices_via_softmax(eta=self.eta, temp=self.temp)
            # self.psi = project_to_vertices_via_softmax(λ=self.lam)
            self.lam = self.eta / self.temp
            self.psi = tf.math.softmax(self.lam, axis=1)
            self.psi = tf.reshape(self.psi, shape=self.psi.numpy().shape + (1,))
        else:
            uniform_sample = tf.random.uniform(shape=(self.batch_size, self.sample_size),
                                               minval=0, maxval=1)
            self.lam, self.psi = project_to_vertices_via_random_jump(
                eta=self.eta, temp=self.temp, uniform_sample=uniform_sample,
                random_jump_threshold=self.random_jump_threshold)

    def get_eta_and_n_required(self, kappa):
        self.perform_truncation_via_threshold(vector=kappa)
        self.eta = kappa[:, :self.n_required, :]


class SBDist(LogitDist):

    def __init__(self, mu: tf.Tensor, xi: tf.Tensor, sample_size: int = 1, noise_type: str = 'normal',
                 temp: tf.Tensor = tf.constant(0.1, dtype=tf.float32), threshold: float = 0.99):
        super().__init__(mu=mu, xi=xi, sample_size=sample_size,
                         noise_type=noise_type, temp=temp, threshold=threshold)

        self.run_iteratively = False
        self.log_jac = tf.constant(0., dtype=tf.float32)
        self.lower = np.zeros(shape=(self.categories_n - 1, self.categories_n - 1))
        self.upper = np.zeros(shape=(self.categories_n - 1, self.categories_n - 1))

    def compute_log_sb_dist(self) -> tf.Tensor:
        log_q_psi = compute_log_sb_dist(lam=self.lam, kappa=self.kappa, sigma=self.sigma,
                                        epsilon=self.epsilon, log_jac=self.log_jac, temp=self.temp)
        return log_q_psi

    def get_eta_and_n_required(self, kappa):
        if self.run_iteratively:
            self.eta, self.log_jac = self.compute_sb_and_log_jac_iteratively(κ=kappa)
        else:
            self.lower, self.upper = generate_lower_and_upper_triangular_matrices_for_sb(
                categories_n=self.categories_n, lower=self.lower, upper=self.upper,
                batch_size=self.batch_size, sample_size=self.sample_size)
            self.eta, self.log_jac = self.perform_stick_break_and_compute_log_jac(kappa=kappa)

    def perform_stick_break_and_compute_log_jac(self, kappa: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        accumulated_prods = accumulate_one_minus_kappa_products_for_sb(kappa=kappa, lower=self.lower,
                                                                       upper=self.upper)
        ς = 1.e-20
        eta = kappa * accumulated_prods
        self.perform_truncation_via_threshold(vector=eta)
        log_jac = -tf.reduce_sum(tf.math.log(accumulated_prods[:, :self.n_required, :] + ς), axis=1)
        return eta[:, :self.n_required, :], log_jac

    def compute_sb_and_log_jac_iteratively(self, κ):
        η, log_jac = iterative_sb_and_jac(κ=κ)
        self.perform_truncation_via_threshold(η)
        return η[:, :self.n_required, :], log_jac


class ExpGSDist(Distributions):

    def __init__(self, log_pi: tf.Tensor, sample_size: int = 1, noise_type: str = 'normal',
                 temp: tf.Tensor = tf.constant(0.1, dtype=tf.float32)):
        super().__init__(batch_size=log_pi.shape[0], categories_n=log_pi.shape[1], sample_size=sample_size,
                         noise_type=noise_type, temp=temp, num_of_vars=log_pi.shape[3])
        self.log_pi = log_pi
        self.log_psi = tf.constant(value=0., dtype=tf.float32)

    def do_reparameterization_trick(self):
        ς = 1.e-20
        log_pi_broad = self.broadcast_params_to_sample_size(params=[self.log_pi])[0]
        uniform = tf.random.uniform(shape=log_pi_broad.shape)
        gumbel_sample = -tf.math.log(-tf.math.log(uniform + ς) + ς)
        y = (log_pi_broad + gumbel_sample) / self.temp
        self.log_psi = y - tf.math.reduce_logsumexp(y, axis=1, keepdims=True)
        self.psi = tf.math.softmax(logits=y, axis=1)


# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Distribution functions
# ===========================================================================================================
def compute_log_sb_dist(lam, kappa, sigma, epsilon, log_jac, temp: tf.Tensor):
    log_q_lam = compute_log_logit_dist(lam=lam, kappa=kappa, sigma=sigma, epsilon=epsilon, temp=temp)
    log_q_psi = log_q_lam + log_jac
    return log_q_psi


def compute_log_logit_dist(lam, kappa, sigma, epsilon, temp: tf.Tensor):
    n_required = epsilon.shape[1]
    log_q_kappa = compute_log_logit_normal(epsilon=epsilon, sigma=sigma, kappa=kappa)
    log_q_psi = log_q_kappa + (n_required * tf.math.log(temp) + temp * tf.math.reduce_sum(lam, axis=1))
    return log_q_psi


def compute_log_logit_dist_projection(kappa, sigma, epsilon, temp: tf.Tensor):
    n_required = epsilon.shape[1]
    log_q_kappa = compute_log_logit_normal(epsilon=epsilon, sigma=sigma, kappa=kappa)
    log_q_psi = log_q_kappa - n_required * tf.math.log(temp)
    return log_q_psi


def compute_log_logit_normal(epsilon, sigma, kappa) -> tf.Tensor:
    log_norm_cons = compute_log_logit_normal_normalizing_constant(sigma, kappa)
    log_exp_sum = -(tf.constant(value=0.5, dtype=tf.float32) * tf.reduce_sum(epsilon ** 2, axis=1))

    log_q_kappa = log_norm_cons + log_exp_sum
    return log_q_kappa


def compute_log_logit_normal_normalizing_constant(sigma, kappa) -> tf.Tensor:
    math_pi = 3.141592653589793
    ς = 1.e-20
    n_required = kappa.shape[1]

    constant_term = -n_required / 2 * tf.math.log(2. * math_pi)
    sigma_term = -tf.reduce_sum(tf.math.log(sigma + ς), axis=1)
    kappa_term = -(tf.reduce_sum(tf.math.log(kappa + ς), axis=1) +
                   tf.reduce_sum(tf.math.log(1 - kappa + ς), axis=1))

    log_norm_const = constant_term + sigma_term + kappa_term
    return log_norm_const


def compute_log_gs_dist(psi: tf.Tensor, logits: tf.Tensor, temp: tf.Tensor) -> tf.Tensor:
    n_required = tf.constant(value=psi.shape[1], dtype=tf.float32)
    ς = tf.constant(1.e-20)

    log_const = tf.math.lgamma(n_required) + (n_required - 1) * tf.math.log(temp)
    log_sum = tf.reduce_sum(logits - (temp + tf.constant(1.)) * tf.math.log(psi + ς), axis=1)
    log_norm = - n_required * tf.math.log(tf.reduce_sum(tf.math.exp(logits) / psi ** temp, axis=1) + ς)

    log_p_concrete = log_const + log_sum + log_norm
    return log_p_concrete


def compute_log_exp_gs_dist(log_psi: tf.Tensor, logits: tf.Tensor, temp: tf.Tensor) -> tf.Tensor:
    categories_n = tf.constant(log_psi.shape[1], dtype=tf.float32)
    log_cons = tf.math.lgamma(categories_n) + (categories_n - 1) * tf.math.log(temp)
    aux = logits - temp * log_psi
    log_sums = tf.math.reduce_sum(aux, axis=1) - categories_n * tf.math.reduce_logsumexp(aux, axis=1)
    log_exp_gs_dist = log_cons + log_sums
    return log_exp_gs_dist


# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Optimization functions for the Expectation Minimization Loss
# ===========================================================================================================
def compute_loss(params: List[tf.Tensor], temp: tf.Tensor, probs: tf.Tensor, dist_type: str = 'sb',
                 sample_size: int = 1, threshold: float = 0.99, run_iteratively=False, run_kl=True):
    chosen_dist = select_chosen_distribution(dist_type=dist_type, params=params, temp=temp,
                                             sample_size=sample_size, threshold=threshold,
                                             run_iteratively=run_iteratively)

    chosen_dist.do_reparameterization_trick()
    psi_mean = tf.reduce_mean(chosen_dist.psi, axis=[0, 2, 3])
    if run_kl:
        loss = psi_mean * (tf.math.log(psi_mean) - tf.math.log(probs[:chosen_dist.n_required] + 1.e-20))
        loss = tf.reduce_sum(loss)
    else:
        loss = tf.reduce_sum((psi_mean - probs[:chosen_dist.n_required]) ** 2)
    return loss, chosen_dist.n_required


def compute_gradients(params, temp: tf.Tensor, probs: tf.Tensor, run_kl=True,
                      dist_type: str = 'sb', sample_size: int = 1, run_iteratively=False,
                      threshold: float = 0.99) -> Tuple[tf.Tensor, tf.Tensor, int]:
    with tf.GradientTape() as tape:
        loss, n_required = compute_loss(params=params, temp=temp, probs=probs, sample_size=sample_size,
                                        threshold=threshold, dist_type=dist_type, run_kl=run_kl,
                                        run_iteratively=run_iteratively)
        gradient = tape.gradient(target=loss, sources=params)
    return gradient, loss, n_required


def apply_gradients(optimizer: tf.keras.optimizers, gradients: tf.Tensor, variables):
    optimizer.apply_gradients(zip(gradients, variables))


# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Utils
# ===========================================================================================================
def retrieve_transformations_up_to_kappa(mu_broad: tf.Tensor, xi_broad: tf.Tensor, epsilon: tf.Tensor):
    sigma = convert_ξ_to_σ(ξ=xi_broad)
    delta = mu_broad + sigma * epsilon
    kappa = tf.math.sigmoid(delta)
    return sigma, delta, kappa


def convert_ξ_to_σ(ξ: tf.Tensor):
    σ = tf.math.exp(ξ)
    return σ


@tf.function
def project_to_vertices_via_softmax(λ):
    ς = 1.e-25
    λ_i_λ_j = λ - tf.math.reduce_max(λ, axis=1, keepdims=True)
    exp_λ = tf.math.exp(λ_i_λ_j)
    norm_λ = tf.math.reduce_sum(exp_λ, axis=1, keepdims=True)
    ψ_plus = exp_λ / (norm_λ + ς)
    return ψ_plus


@tf.function
def project_to_vertices_via_softmax_pp(lam):
    offset = 1.e-1
    lam_i_lam_max = lam - tf.math.reduce_max(lam, axis=1, keepdims=True)
    exp_lam = tf.math.exp(lam_i_lam_max)
    norm_lam = tf.math.reduce_sum(exp_lam, axis=1, keepdims=True)
    aux = exp_lam / (norm_lam + offset)

    psi_plus = (1 - tf.math.reduce_sum(aux, axis=1, keepdims=True))
    psi = tf.concat(values=[aux, psi_plus], axis=1)

    return psi


def project_to_vertices_via_softplus(lam):
    ς = 1.e-20
    normalized_psi = tf.math.reduce_sum(tf.math.softplus(lam), axis=1, keepdims=True) + ς
    psi = tf.math.softplus(lam) / normalized_psi
    return psi


def project_to_vertices_via_random_jump(eta, temp: tf.Tensor, uniform_sample, random_jump_threshold):
    λ = eta
    batch_size, categories_n, sample_size = eta.shape
    ψ = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=(categories_n, sample_size))
    # noinspection PyTypeChecker
    projection = temp * eta + (1. - temp) * project_into_simplex(eta)
    for i in tf.range(batch_size):
        if uniform_sample[i] <= random_jump_threshold:
            ψ = ψ.write(index=i, value=eta[i, :, :])
        else:
            ψ = ψ.write(index=i, value=projection[i, :, :])
    return λ, ψ.stack()


def project_into_simplex(vector: tf.Tensor):
    batch_size, n_required, sample_size = vector.shape
    projection = np.zeros(shape=(batch_size, n_required, sample_size))

    argmax_loc = np.argmax(vector.numpy(), axis=1)
    for sample in range(sample_size):
        for batch in range(batch_size):
            projection[batch, argmax_loc[batch, sample], sample] = 1.

    projection = tf.constant(value=projection, dtype=tf.float32)
    return projection


def accumulate_one_minus_kappa_products_for_sb(kappa: tf.Tensor, lower, upper) -> tf.Tensor:
    forget_last = -1
    one = tf.constant(value=1., dtype=tf.float32)

    diagonal_kappa = tf.linalg.diag(tf.transpose(one - kappa[:, :forget_last, :], perm=[0, 2, 1]))
    accumulation = tf.transpose(tf.tensordot(lower, diagonal_kappa, axes=[[1], [2]]),
                                perm=[1, 0, 3, 2])
    accumulation_w_ones = accumulation + upper
    cumprod = tf.math.reduce_prod(input_tensor=accumulation_w_ones, axis=2)
    return cumprod


def generate_lower_and_upper_triangular_matrices_for_sb(categories_n, lower, upper,
                                                        batch_size, sample_size):
    zeros_row = np.zeros(shape=categories_n - 1)

    for i in range(categories_n - 1):
        for j in range(categories_n - 1):
            if i > j:
                lower[i, j] = 1
            elif i == j:
                lower[i, j] = 1
                upper[i, j] = 1
            else:
                upper[i, j] = 1

    lower = np.vstack([zeros_row, lower])
    upper = np.vstack([upper, zeros_row])

    upper = np.broadcast_to(upper, shape=(batch_size, categories_n, categories_n - 1))
    upper = np.reshape(upper, newshape=(batch_size, categories_n, categories_n - 1, 1))
    upper = np.broadcast_to(upper, shape=(batch_size, categories_n, categories_n - 1, sample_size))
    lower = tf.constant(value=lower, dtype=tf.float32)  # no reshape needed
    upper = tf.constant(value=upper, dtype=tf.float32)
    return lower, upper


@tf.function
def iterative_sb_and_jac(κ):
    batch_size, max_size, samples_n = κ.shape
    ς = 1.e-20
    η = tf.TensorArray(dtype=tf.float32, size=max_size, element_shape=(batch_size, samples_n),
                       clear_after_read=True)
    η = η.write(index=0, value=κ[:, 0, :])
    cumsum = tf.identity(κ[:, 0, :])
    next_cumsum = tf.identity(κ[:, 1, :] * (1 - κ[:, 0, :]) + κ[:, 0, :])
    jac_sum = tf.constant(value=0., dtype=tf.float32, shape=(batch_size, samples_n))
    max_iter = tf.constant(value=max_size - 1, dtype=tf.int32)
    for i in tf.range(1, max_iter):
        η = η.write(index=i, value=κ[:, i, :] * (1. - cumsum))
        jac_sum += tf.math.log(1. - cumsum + ς)
        cumsum += κ[:, i, :] * (1. - cumsum)
        next_cumsum += κ[:, i + 1, :] * (1. - next_cumsum)

    η = η.write(index=max_size - 1, value=κ[:, max_size - 1, :] * (1. - cumsum))
    jac_sum += tf.math.log(1. - cumsum + ς)
    return tf.transpose(η.stack(), perm=[1, 0, 2]), -jac_sum


@tf.function
def compute_log_jac(κ):
    batch_size, n_required, samples_n = κ.shape
    cumsum = tf.identity(κ[:, 0, :])
    jac_sum = tf.constant(value=0., dtype=tf.float32, shape=(batch_size, samples_n))
    max_iter = tf.constant(value=n_required, dtype=tf.int32)
    for i in tf.range(1, max_iter):
        jac_sum += tf.math.log(1. - cumsum + 1.e-20)
        cumsum += κ[:, i, :] * (1. - cumsum)
    return -jac_sum


def generate_sample(sample_size: int, params, dist_type: str, temp, threshold: float = 0.99,
                    output_one_hot=False):
    chosen_dist = select_chosen_distribution(dist_type=dist_type, threshold=threshold,
                                             params=params, temp=temp, sample_size=sample_size)
    categories_n = params[0].shape[1]
    chosen_dist.do_reparameterization_trick()
    if output_one_hot:
        vector = np.zeros(shape=(1, categories_n, sample_size, 1))
        n_required = chosen_dist.psi.shape[1]
        vector[:, :n_required, :, :] = chosen_dist.psi.numpy()
        return vector
    else:
        sample = np.argmax(chosen_dist.psi.numpy(), axis=1)
        return sample


def select_chosen_distribution(dist_type: str, params, temp=tf.constant(0.1, dtype=tf.float32),
                               sample_size: int = 1, threshold: float = 0.99, run_iteratively=False):
    if dist_type == 'logit':
        # noinspection PyTypeChecker
        mu, xi = params
        chosen_dist = LogitDist(mu=mu, xi=xi, temp=temp, sample_size=sample_size, threshold=threshold)
    elif dist_type == 'ExpGS':
        pi = params[0]
        chosen_dist = ExpGSDist(log_pi=pi, temp=temp, sample_size=sample_size)
    elif dist_type == 'IsoGauSoftMax':
        mu = params[0]
        chosen_dist = IsoGauSoftMax(mu=mu, temp=temp, sample_size=sample_size)
    elif dist_type == 'GauSoftMax':
        mu, xi, = params
        chosen_dist = GaussianSoftmaxDist(mu=mu, xi=xi, temp=temp, sample_size=sample_size)
    elif dist_type == 'GauSoftPlus':
        mu, xi, = params
        chosen_dist = GaussianSoftPlus(mu=mu, xi=xi, temp=temp, sample_size=sample_size)
    elif dist_type == 'Cauchy':
        mu, xi, = params
        chosen_dist = CauchySoftmaxDist(mu=mu, xi=xi, temp=temp, sample_size=sample_size)
    elif dist_type == 'sb':
        # noinspection PyTypeChecker
        mu, xi = params
        chosen_dist = SBDist(mu=mu, xi=xi, temp=temp, sample_size=sample_size, threshold=threshold)
        if run_iteratively:
            chosen_dist.run_iteratively = True
    else:
        raise RuntimeError

    return chosen_dist


def generate_samples_mp(total_samples, params, dist_type, threshold, temp, pool, output_one_hot=False):
    # TODO: correct this function
    func = partial(generate_sample, params=params, threshold=threshold,
                   dist_type=dist_type, temp=temp, output_one_hot=output_one_hot)
    # noinspection PyTypeChecker
    sb_samples = np.array(pool.map(func=func, iterable=[b for b in range(total_samples)]))

    return sb_samples
# ===========================================================================================================

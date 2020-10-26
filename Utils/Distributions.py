import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, List
from os import environ as os_env
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Distributions:

    def __init__(self, batch_size: int, categories_n: int, sample_size: int = 1,
                 num_of_vars: int = 1, noise_type: str = 'normal',
                 temp: tf.Tensor = tf.constant(0.1, dtype=tf.float32)):

        self.noise_type = noise_type
        self.temp = temp
        self.batch_size = batch_size
        self.categories_n = categories_n
        self.sample_size = sample_size
        self.num_of_vars = num_of_vars

        self.n_required = categories_n
        self.kappa = tf.constant(0., dtype=temp.dtype)
        self.lam = tf.constant(0., dtype=temp.dtype)
        self.log_psi = tf.constant(0., dtype=temp.dtype)
        self.psi = tf.constant(0., dtype=temp.dtype)

    def broadcast_params_to_sample_size(self, params: list):
        params_broad = []
        for param in params:
            shape = (self.batch_size, self.categories_n,
                     self.sample_size, self.num_of_vars)
            param_w_samples = tf.broadcast_to(input=param, shape=shape)
            params_broad.append(param_w_samples)
        return params_broad

    def sample_noise(self, tf_shape):
        if self.noise_type == 'normal':
            epsilon = tf.random.normal(shape=tf_shape)
        elif self.noise_type == 'trunc_normal':
            epsilon = tf.random.truncated_normal(shape=tf_shape)
        else:
            RuntimeError
        return epsilon


class IGR_I(Distributions):
    def __init__(self, mu, xi, temp, sample_size=1, noise_type='normal'):
        super().__init__(batch_size=mu.shape[0], categories_n=mu.shape[1],
                         sample_size=sample_size,
                         noise_type=noise_type, temp=temp, num_of_vars=mu.shape[3])

        self.mu = mu
        self.xi = xi

    def generate_sample(self):
        mu_broad, xi_broad = self.broadcast_params_to_sample_size(
            params=[self.mu, self.xi])
        epsilon = tf.random.normal(shape=mu_broad.shape, dtype=mu_broad.dtype)
        sigma_broad = tf.math.exp(xi_broad)
        # sigma_broad = tf.math.softplus(xi_broad + tf.constant(1.))
        self.kappa = mu_broad + sigma_broad * epsilon
        self.lam = self.transform()
        self.log_psi = self.lam - \
            tf.math.reduce_logsumexp(self.lam, axis=1, keepdims=True)
        self.psi = project_to_vertices_via_softmax_pp(self.lam / self.temp)

        # self.psi = tf.math.softmax(self.lam / self.temp, axis=1)
        # Check in train_model for the hyperparameter addition of 1 category
        # Look for the line:
        # hyper_copy['latent_discrete_n'] += 1

    def transform(self):
        lam = self.kappa
        return lam


class IGR_Planar(IGR_I):
    def __init__(self, mu, xi, temp, planar_flow, sample_size=1, noise_type='normal'):
        super().__init__(mu, xi, temp, sample_size, noise_type)
        self.planar_flow = planar_flow

    def transform(self):
        lam = (self.planar_flow(self.kappa))
        return lam


class IGR_SB(IGR_I):

    def __init__(self, mu, xi, temp, sample_size=1, noise_type='normal', threshold=0.99,
                 run_iteratively=False):
        super().__init__(mu, xi, temp, sample_size, noise_type)

        self.threshold = threshold
        self.truncation_option = 'quantile'
        self.quantile = 70
        self.run_iteratively = run_iteratively

    def transform(self):
        eta = tf.math.sigmoid(self.kappa)
        lam = self.apply_stick_break(eta)
        return lam

    def apply_stick_break(self, kappa):
        eta = iterative_sb(
            kappa) if self.run_iteratively else self.perform_matrix_sb(kappa)
        self.perform_truncation_via_threshold(vector=eta)
        return eta[:, :self.n_required, :, :]

    def perform_matrix_sb(self, kappa):
        lower, upper = self.construct_matrices_for_sb()
        accumulated_prods = accumulate_one_minus_kappa_prods(kappa, lower, upper)
        eta = kappa * accumulated_prods
        return eta

    def construct_matrices_for_sb(self):
        lower, upper = generate_lower_and_upper_triangular_matrices(self.categories_n)
        lower, upper = broadcast_matrices_to_shape(lower, upper, self.batch_size,
                                                   self.categories_n, self.sample_size,
                                                   self.num_of_vars)
        return lower, upper

    def perform_truncation_via_threshold(self, vector):
        vector_cumsum = tf.math.cumsum(x=vector, axis=1)
        res = get_arrays_that_make_it(vector_cumsum, tf.constant(self.threshold))
        res += 1
        # lower_bound = tf.constant(3, dtype=tf.int32)
        # batch_ind = int(tfp.stats.percentile(res, q=self.quantile))
        # self.n_required = tf.maximum(lower_bound, batch_ind)
        self.n_required = int(tfp.stats.percentile(res, q=self.quantile))


class IGR_SB_Finite(IGR_SB):
    def __init__(self, mu, xi, temp, sample_size=1, noise_type='normal'):
        super().__init__(mu=mu, xi=xi, temp=temp, sample_size=sample_size,
                         noise_type=noise_type)

    def apply_stick_break(self, kappa):
        eta = iterative_sb(
            kappa) if self.run_iteratively else self.perform_matrix_sb(kappa)
        return eta


class GS(Distributions):
    def __init__(self, log_pi, temp, sample_size: int = 1):
        super().__init__(batch_size=log_pi.shape[0], categories_n=log_pi.shape[1],
                         sample_size=sample_size,
                         temp=temp, num_of_vars=log_pi.shape[3])
        self.log_pi = log_pi

    def generate_sample(self):
        offset = 1.e-20
        log_pi_broad = self.broadcast_params_to_sample_size(params=[self.log_pi])[0]
        uniform = tf.random.uniform(shape=log_pi_broad.shape, dtype=log_pi_broad.dtype)
        gumbel_sample = -tf.math.log(-tf.math.log(uniform + offset) + offset)
        self.lam = (log_pi_broad + gumbel_sample) / self.temp
        self.log_psi = self.lam - \
            tf.math.reduce_logsumexp(self.lam, axis=1, keepdims=True)
        self.psi = tf.math.softmax(logits=self.lam, axis=1)


def compute_log_gs_dist(psi: tf.Tensor, logits: tf.Tensor, temp: tf.Tensor) -> tf.Tensor:
    n_required = tf.constant(value=psi.shape[1], dtype=tf.float32)
    offset = tf.constant(1.e-20)

    log_const = tf.math.lgamma(n_required) + (n_required - 1) * tf.math.log(temp)
    log_sum = tf.reduce_sum(logits - (temp + tf.constant(1.))
                            * tf.math.log(psi + offset), axis=1)
    log_norm = - n_required * \
        tf.math.log(tf.reduce_sum(tf.math.exp(logits) / psi ** temp, axis=1) + offset)

    log_p_concrete = log_const + log_sum + log_norm
    return log_p_concrete


def compute_log_exp_gs_dist(log_psi: tf.Tensor, logits: tf.Tensor,
                            temp: tf.Tensor) -> tf.Tensor:
    categories_n = tf.constant(log_psi.shape[1], dtype=log_psi.dtype)
    log_cons = tf.math.lgamma(categories_n) + (categories_n - 1) * tf.math.log(temp)
    aux = logits - temp * log_psi
    log_sums = tf.math.reduce_sum(aux, axis=1)
    log_sums -= categories_n * tf.math.reduce_logsumexp(aux, axis=1)
    log_exp_gs_dist = log_cons + log_sums
    return log_exp_gs_dist


def compute_loss(params: List[tf.Tensor], temp: tf.Tensor, probs: tf.Tensor,
                 dist_type: str = 'sb',
                 sample_size: int = 1, threshold: float = 0.99,
                 run_iteratively=False, run_kl=True,
                 planar_flow: str = None):
    chosen_dist = select_chosen_distribution(dist_type=dist_type, params=params,
                                             temp=temp,
                                             sample_size=sample_size,
                                             threshold=threshold,
                                             run_iteratively=run_iteratively,
                                             planar_flow=planar_flow)

    chosen_dist.generate_sample()
    psi_mean = tf.reduce_mean(chosen_dist.psi, axis=[0, 2, 3])
    if run_kl:
        if dist_type == 'GS':
            loss = psi_mean * (tf.math.log(psi_mean) -
                               tf.math.log(probs[:chosen_dist.n_required]))
        else:
            loss = psi_mean * (tf.math.log(psi_mean) -
                               tf.math.log(probs[:chosen_dist.n_required + 1]))
        loss = tf.reduce_sum(loss)
    else:
        loss = tf.reduce_sum((psi_mean - probs[:chosen_dist.n_required + 1]) ** 2)
    return loss, chosen_dist.n_required


def compute_gradients(params, temp: tf.Tensor, probs: tf.Tensor, run_kl=True,
                      dist_type: str = 'sb', sample_size: int = 1,
                      run_iteratively=False, threshold: float = 0.99,
                      planar_flow: str = None) -> Tuple[tf.Tensor, tf.Tensor, int]:
    with tf.GradientTape() as tape:
        loss, n_required = compute_loss(params=params, temp=temp, probs=probs,
                                        sample_size=sample_size,
                                        threshold=threshold, dist_type=dist_type,
                                        run_kl=run_kl,
                                        run_iteratively=run_iteratively,
                                        planar_flow=planar_flow)
        gradient = tape.gradient(target=loss, sources=params)
    return gradient, loss, n_required


def apply_gradients(optimizer: tf.keras.optimizers, gradients: tf.Tensor, variables):
    optimizer.apply_gradients(zip(gradients, variables))


def project_to_vertices(z, categories_n):
    one_hot = tf.transpose(tf.one_hot(tf.argmax(tf.stop_gradient(z), axis=1),
                                      depth=categories_n, dtype=z.dtype),
                           perm=[0, 3, 1, 2])
    return one_hot


def project_to_vertices_via_softmax_pp(lam):
    delta = tf.constant(1., dtype=lam.dtype)
    one = tf.constant(1.0, dtype=lam.dtype)

    lam_max = tf.math.reduce_max(lam, axis=1, keepdims=True)
    exp_lam = tf.math.exp(lam - lam_max)
    sum_exp_lam = tf.math.reduce_sum(exp_lam, axis=1, keepdims=True)
    psi = exp_lam / (sum_exp_lam + delta * tf.math.exp(-lam_max))

    extra_cat = (one - tf.math.reduce_sum(psi, axis=1, keepdims=True))
    psi = tf.concat(values=[psi, extra_cat], axis=1)

    return psi


def get_arrays_that_make_it(vector_cumsum, threshold):
    batch_n, categories_n, sample_size, _ = vector_cumsum.shape
    result = tf.TensorArray(tf.int64, size=batch_n * sample_size)
    i = tf.constant(0)
    for b in tf.range(batch_n):
        for s in tf.range(sample_size):
            thres = tf.where(vector_cumsum[b, :, s, 0] <= threshold)
            max_val = tf.math.reduce_max(thres, axis=0)[0]
            result = result.write(i, max_val)
            i += 1
    return result.stack()


def accumulate_one_minus_kappa_prods(kappa, lower, upper):
    forget_last = -1
    one = tf.constant(value=1., dtype=tf.float32)

    diagonal_kappa = tf.linalg.diag(tf.transpose(
        one - kappa[:, :forget_last, :, :], perm=[0, 2, 3, 1]))
    accumulation = tf.transpose(tf.tensordot(lower, diagonal_kappa,
                                             axes=[[1], [3]]), perm=[1, 0, 4, 2, 3])
    accumulation_w_ones = accumulation + upper
    cumprod = tf.math.reduce_prod(input_tensor=accumulation_w_ones, axis=2)
    return cumprod


def generate_lower_and_upper_triangular_matrices(categories_n):
    lower = np.zeros(shape=(categories_n - 1, categories_n - 1))
    upper = np.zeros(shape=(categories_n - 1, categories_n - 1))
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

    return lower, upper


def broadcast_matrices_to_shape(lower, upper, batch_size, categories_n,
                                sample_size, num_of_vars):
    upper = np.broadcast_to(upper, shape=(batch_size, categories_n, categories_n - 1))
    upper = np.reshape(upper, newshape=(
        batch_size, categories_n, categories_n - 1, 1, 1))
    upper = np.broadcast_to(upper, shape=(batch_size, categories_n,
                                          categories_n - 1, sample_size, 1))
    upper = np.broadcast_to(upper, shape=(batch_size, categories_n,
                                          categories_n - 1, sample_size, num_of_vars))

    lower = tf.constant(value=lower, dtype=tf.float32)  # no reshape needed
    upper = tf.constant(value=upper, dtype=tf.float32)
    return lower, upper


def iterative_sb(kappa):
    batch_size, max_size, sample_size, num_of_vars = kappa.shape
    eta = tf.TensorArray(dtype=kappa.dtype, size=max_size + 1,
                         element_shape=(batch_size, sample_size, num_of_vars))
    eta = eta.write(index=0, value=kappa[:, 0, :, :])
    cumsum = tf.identity(kappa[:, 0, :, :])  # copy contents
    for i in range(1, max_size):
        eta = eta.write(index=i, value=kappa[:, i, :, :] * (1. - cumsum))
        cumsum += kappa[:, i, :, :] * (1. - cumsum)
    eta = eta.write(index=max_size, value=1. - cumsum)
    return tf.transpose(eta.stack(), perm=[1, 0, 2, 3])


def generate_sample(sample_size: int, params, dist_type: str, temp,
                    threshold: float = 0.99,
                    output_one_hot=False, planar_flow=None):
    chosen_dist = select_chosen_distribution(dist_type=dist_type, threshold=threshold,
                                             params=params, temp=temp,
                                             sample_size=sample_size,
                                             planar_flow=planar_flow)
    chosen_dist.generate_sample()
    if output_one_hot:
        return chosen_dist.psi.numpy()
    else:
        sample = np.argmax(chosen_dist.psi.numpy(), axis=1)[0, 0, 0]
        return sample


def select_chosen_distribution(dist_type: str, params,
                               temp=tf.constant(0.1, dtype=tf.float32),
                               sample_size: int = 1, threshold: float = 0.99,
                               run_iteratively=False, planar_flow=None):
    if dist_type == 'IGR_SB':
        mu, xi = params
        chosen_dist = IGR_SB(mu=mu, xi=xi, temp=temp,
                             sample_size=sample_size, threshold=threshold)
        if run_iteratively:
            chosen_dist.run_iteratively = True
    elif dist_type == 'IGR_SB_Finite':
        mu, xi = params
        chosen_dist = IGR_SB_Finite(mu=mu, xi=xi, temp=temp, sample_size=sample_size)
        if run_iteratively:
            chosen_dist.run_iteratively = True
    elif dist_type == 'GS':
        pi = params[0]
        chosen_dist = GS(log_pi=pi, temp=temp, sample_size=sample_size)
    elif dist_type == 'IGR_I':
        mu, xi, = params
        chosen_dist = IGR_I(mu=mu, xi=xi, temp=temp, sample_size=sample_size)
    elif dist_type == 'IGR_Planar':
        mu, xi, = params
        chosen_dist = IGR_Planar(mu=mu, xi=xi, temp=temp,
                                 planar_flow=planar_flow, sample_size=sample_size)
    else:
        raise RuntimeError

    return chosen_dist


def compute_log_gauss_grad(z, mu, sigma):
    grad_mu = compute_log_gauss_grad_mu(z, mu, sigma)
    grad_sigma = compute_log_gauss_grad_sigma(z, mu, sigma)
    grads = [grad_mu, grad_sigma]
    return grads


def compute_log_gauss_grad_mu(z, mu, sigma):
    grad = (z - mu) / (sigma ** 2)
    return grad


def compute_log_gauss_grad_sigma(z, mu, sigma):
    grad = - 1 / sigma
    grad += (z - mu) ** 2 / sigma ** 3
    return grad


def compute_igr_log_probs(mu, sigma):
    log_integral_probs = compute_log_probs_via_quad(mu, sigma)
    log_last_prob = compute_log_last_prob(mu, sigma)
    log_probs = tf.concat([log_integral_probs, log_last_prob], axis=1)
    return log_probs


def compute_log_last_prob(mu, sigma):
    gaussian = tfp.distributions.Normal(loc=tf.constant(0., dtype=mu.dtype),
                                        scale=tf.constant(1., dtype=mu.dtype))
    cdf_broad = gaussian.log_cdf((-mu) / sigma)
    log_last_prob = tf.math.reduce_sum(cdf_broad, axis=1, keepdims=True)
    return log_last_prob


def compute_log_probs_via_quad(mu, sigma):
    # w = [8.62207055355942e-02, 1.85767318955695e-01,
    #      2.35826124129815e-01, 2.05850326841520e-01,
    #      1.19581170615297e-01, 4.31443275880520e-02,
    #      8.86764989474414e-03, 9.27141875082127e-04,
    #      4.15719321667468e-05, 5.86857646837617e-07, 1.22714513994286e-09]
    # y = [3.38393212320868e-02, 1.73955727711686e-01,
    #      4.10873440975301e-01, 7.26271784264131e-01,
    #      1.10386324647012e+00, 1.53229503458121e+00,
    #      2.00578290247431e+00, 2.52435214152551e+00,
    #      3.09535170987551e+00, 3.73947860994972e+00, 4.51783596719327e+00]
    w = [5.54433663102343e-02, 1.24027738987730e-01,
         1.75290943892075e-01, 1.91488340747342e-01,
         1.63473797144070e-01, 1.05937637278492e-01,
         5.00270211534535e-02, 1.64429690052673e-02,
         3.57320421428311e-03, 4.82896509305201e-04,
         3.74908650266318e-05, 1.49368411589636e-06,
         2.55270496934465e-08, 1.34217679136316e-10,
         9.56227446736465e-14]
    y = [2.16869474675590e-02, 1.12684220347775e-01,
         2.70492671421899e-01, 4.86902370381935e-01,
         7.53043683072978e-01, 1.06093100362236e+00,
         1.40425495820363e+00, 1.77864637941183e+00,
         2.18170813144494e+00, 2.61306084533352e+00,
         3.07461811380851e+00, 3.57140815113714e+00,
         4.11373608977209e+00, 4.72351306243148e+00, 5.46048893678335e+00]
    w = reshape_for_quad(w, mu.shape, mu.dtype)
    y = reshape_for_quad(y, mu.shape, mu.dtype)

    log_h_f = compute_log_h_f(y, mu, sigma)
    log_integral = tf.math.reduce_logsumexp(tf.math.log(w) + log_h_f, axis=-1)
    return log_integral


def reshape_for_quad(v, shape, dtype):
    v = tf.constant(v, dtype=dtype)
    # v = tf.reshape(v, (1, 1, 1, 1, 11))
    # v = tf.broadcast_to(v, shape + (11,))
    v = tf.reshape(v, (1, 1, 1, 1, 15))
    v = tf.broadcast_to(v, shape + (15,))
    return v


def compute_log_h_f(y, mu, sigma):
    mu_expanded = tf.expand_dims(mu, -1)
    sigma_expanded = tf.expand_dims(sigma, -1)
    gaussian = tfp.distributions.Normal(loc=tf.constant(0., dtype=mu.dtype),
                                        scale=tf.constant(1., dtype=mu.dtype))

    t = tf.math.sqrt(2 * sigma_expanded ** 2) * y
    cons = tf.constant(3.141592653589793, dtype=mu.dtype) ** (-0.5)
    exp_term = (1 / (2 * sigma_expanded ** 2)) * (2 * mu_expanded * t - mu_expanded ** 2)
    cdf_broad = gaussian.log_cdf((t - mu_expanded) / sigma_expanded)
    # TODO: see if I should stop the gradient of cdf_broad
    cdf_term = tf.math.reduce_sum(cdf_broad, axis=1, keepdims=True) - cdf_broad
    output = tf.math.log(cons) + cdf_term + exp_term
    return output + 1.e-20


def compute_igr_probs(mu, sigma):
    integral = tf.math.exp(compute_log_probs_via_quad(mu, sigma))
    remainder = tf.constant(1., dtype=mu.dtype) - \
        tf.reduce_sum(integral, axis=1, keepdims=True)
    return tf.clip_by_value(tf.concat([integral, remainder], axis=1), 1.e-20, 1.)


def compute_h_f(y, mu, sigma):
    mu_expanded = tf.expand_dims(mu, -1)
    sigma_expanded = tf.expand_dims(sigma, -1)
    gaussian = tfp.distributions.Normal(loc=0., scale=1.)

    t = tf.math.sqrt(tf.constant(2.)) * sigma_expanded * y
    cons = tf.constant(3.141592653589793) ** (-0.5)
    inner_exp = (1 / (2 * sigma_expanded ** 2)) * \
        (2 * mu_expanded * t - mu_expanded ** 2)
    exp_term = tf.math.exp(tf.clip_by_value(inner_exp, -50., 50.))
    denom = gaussian.cdf((t - mu_expanded) / sigma_expanded)
    num = tf.math.reduce_prod(denom, axis=1, keepdims=True)
    output = cons * (num / tf.clip_by_value(denom, 1.e-10, 1.)) * exp_term
    return output

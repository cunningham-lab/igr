import pickle
from typing import Tuple
import tensorflow as tf
from os import environ as os_env
from Utils.Distributions import IGR_I, IGR_Planar, IGR_SB, IGR_SB_Finite, GS, compute_log_exp_gs_dist
from Utils.general import initialize_mu_and_xi_for_logistic
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'


class OptVAE:

    def __init__(self, nets, optimizer, hyper):
        self.nets = nets
        self.optimizer = optimizer
        self.run_closed_form_kl = hyper['run_closed_form_kl']
        self.batch_size = hyper['sample_size']
        self.n_required = hyper['n_required']
        self.sample_size = hyper['sample_size']
        self.num_of_vars = hyper['num_of_discrete_var']
        self.dataset_name = hyper['dataset_name']

        self.run_jv = hyper['run_jv']
        self.gamma = hyper['gamma']
        self.discrete_c = tf.constant(0.)
        self.continuous_c = tf.constant(0.)

    def perform_fwd_pass(self, x):
        params = self.nets.encode(x)
        z = self.reparameterize(params_broad=params)
        x_logit = self.decode(z=z)
        return z, x_logit, params

    def reparameterize(self, params_broad):
        mean, log_var = params_broad
        z = sample_normal(mean=mean, log_var=log_var)
        return z

    def decode(self, z):
        if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
            x_logit = self.decode_gaussian(z=z)
        else:
            x_logit = self.decode_bernoulli(z=z)
        return x_logit

    def decode_bernoulli(self, z):
        z = reshape_and_stack_z(z=z)
        batch_n, sample_size = z.shape[0], z.shape[2]
        x_logit = tf.TensorArray(dtype=tf.float32, size=sample_size,
                                 element_shape=(batch_n,) + self.nets.image_shape)
        for i in tf.range(sample_size):
            x_logit = x_logit.write(index=i, value=self.nets.decode(z[:, :, i])[0])
        x_logit = tf.transpose(x_logit.stack(), perm=[1, 2, 3, 4, 0])
        return x_logit

    def decode_gaussian(self, z):
        z = reshape_and_stack_z(z=z)
        batch_n, sample_size = z.shape[0], z.shape[2]
        mu = tf.TensorArray(dtype=tf.float32, size=sample_size,
                            element_shape=(batch_n,) + self.nets.image_shape)
        xi = tf.TensorArray(dtype=tf.float32, size=sample_size,
                            element_shape=(batch_n,) + self.nets.image_shape)
        for i in tf.range(sample_size):
            z_mu, z_xi = self.nets.decode(z[:, :, i])
            mu = mu.write(index=i, value=z_mu)
            xi = xi.write(index=i, value=z_xi)
        mu = tf.transpose(mu.stack(), perm=[1, 2, 3, 4, 0])
        xi = tf.transpose(xi.stack(), perm=[1, 2, 3, 4, 0])
        x_logit = [mu, xi]
        return x_logit

    @staticmethod
    def compute_kl_elements(z, params_broad, run_closed_form_kl):
        mean, log_var = params_broad
        kl_norm = sample_kl_norm(z_norm=z, mean=mean, log_var=log_var)
        kl_dis = tf.constant(0)
        return kl_norm, kl_dis

    def compute_loss(self, x, x_logit, z, params_broad, run_jv, run_closed_form_kl):
        if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
            log_px_z = compute_log_gaussian_pdf(x=x, x_logit=x_logit)
        else:
            log_px_z = compute_log_bernoulli_pdf(x=x, x_logit=x_logit)
        kl_norm, kl_dis = self.compute_kl_elements(z=z, params_broad=params_broad,
                                                   run_closed_form_kl=run_closed_form_kl)
        kl = kl_norm + kl_dis
        loss = compute_loss(log_px_z=log_px_z, kl_norm=kl_norm, kl_dis=kl_dis,
                            run_jv=run_jv, gamma=self.gamma,
                            discrete_c=self.discrete_c, continuous_c=self.continuous_c)
        output = (loss, tf.reduce_mean(log_px_z), tf.reduce_mean(kl),
                  tf.reduce_mean(kl_norm), tf.reduce_mean(kl_dis))
        return output

    def compute_losses_from_x_wo_gradients(self, x, run_jv, run_closed_form_kl):
        z, x_logit, params_broad = self.perform_fwd_pass(x=x)
        output = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                   run_jv=run_jv, run_closed_form_kl=run_closed_form_kl)
        loss, recon, kl, kl_norm, kl_dis = output
        return loss, recon, kl, kl_norm, kl_dis

    def compute_gradients(self, x) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x)
            output = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                       run_jv=self.run_jv, run_closed_form_kl=self.run_closed_form_kl)
            loss, recon, kl, kl_n, kl_d = output
        gradients = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        return gradients, loss, recon, kl, kl_n, kl_d

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.nets.trainable_variables))

    def monitor_parameter_gradients_at_psi(self, x):
        params = self.nets.encode(x)
        with tf.GradientTape() as tape:
            tape.watch(params[0])
            _ = self.reparameterize(params)
            # noinspection PyUnresolvedReferences
            psi = tf.math.softmax(self.dist.lam, axis=1)
        gradients = tape.gradient(target=psi, sources=params[0])
        gradients_norm = tf.linalg.norm(gradients, axis=1)

        return gradients_norm


class OptExpGS(OptVAE):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.temp = tf.constant(value=hyper['temp'], dtype=tf.float32)
        self.dist = GS(log_pi=tf.constant(1., dtype=tf.float32, shape=(1, 1, 1, 1)), temp=self.temp)
        self.log_psi = tf.constant(1., dtype=tf.float32, shape=(1, 1, 1, 1))

    def reparameterize(self, params_broad):
        mean, log_var, logits = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        self.dist = GS(log_pi=logits, sample_size=self.sample_size, temp=self.temp)
        self.dist.generate_sample()
        z_discrete = self.dist.psi
        self.log_psi = self.dist.log_psi
        self.n_required = z_discrete.shape[1]
        z = [z_norm, z_discrete]
        return z

    def compute_kl_elements(self, z, params_broad, run_closed_form_kl):
        mean, log_var, logits = params_broad
        z_norm, _ = z
        kl_norm = sample_kl_norm(z_norm=z_norm, mean=mean, log_var=log_var)
        kl_dis = sample_kl_exp_gs(log_psi=self.log_psi, log_pi=logits, temp=self.temp)
        return kl_norm, kl_dis


class OptExpGSDis(OptExpGS):
    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)

    def reparameterize(self, params_broad):
        self.dist = GS(log_pi=params_broad[0], sample_size=self.sample_size, temp=self.temp)
        self.dist.generate_sample()
        self.log_psi = self.dist.log_psi
        self.n_required = self.dist.psi.shape[1]
        z_discrete = [self.dist.psi]
        # z_discrete = [gs.log_psi]
        return z_discrete

    def compute_kl_elements(self, z, params_broad, run_closed_form_kl):
        if run_closed_form_kl:
            kl_norm, kl_dis = self.compute_kl_elements_via_closed_cat(params_broad=params_broad)
        else:
            kl_norm, kl_dis = self.compute_kl_elements_via_sample(params_broad=params_broad)
        return kl_norm, kl_dis

    @staticmethod
    def compute_kl_elements_via_closed_cat(params_broad):
        kl_norm = 0.
        kl_dis = calculate_categorical_closed_kl(log_alpha=params_broad[0])
        return kl_norm, kl_dis

    def compute_kl_elements_via_sample(self, params_broad):
        kl_norm = 0.
        kl_dis = sample_kl_exp_gs(log_psi=self.log_psi, log_pi=params_broad[0], temp=self.temp)
        return kl_norm, kl_dis


class OptIGR(OptVAE):
    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.temp = tf.constant(value=hyper['temp'], dtype=tf.float32)
        self.mu_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.xi_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.dist = IGR_I(mu=self.mu_0, xi=self.xi_0, temp=self.temp)
        self.use_continuous = True

    def reparameterize(self, params_broad):
        mean, log_var, mu, xi = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        self.select_distribution(mu, xi)
        self.dist.generate_sample()
        z_discrete = self.dist.psi
        self.n_required = z_discrete.shape[1]
        z = [z_norm, z_discrete]
        return z

    def select_distribution(self, mu, xi):
        self.dist = IGR_I(mu=mu, xi=xi, temp=self.temp)

    def compute_kl_elements(self, z, params_broad, run_closed_form_kl):
        if self.use_continuous:
            mean, log_var, mu_disc, xi_disc = params_broad
            kl_norm = calculate_simple_closed_gauss_kl(mean=mean, log_var=log_var)
        else:
            mu_disc, xi_disc = params_broad
            kl_norm = 0.
        current_batch_n = self.dist.lam.shape[0]
        mu_disc_prior = self.mu_0[:current_batch_n, :, :]
        xi_disc_prior = self.xi_0[:current_batch_n, :, :]
        kl_dis = calculate_general_closed_form_gauss_kl(mean_q=mu_disc, log_var_q=2 * xi_disc,
                                                        mean_p=mu_disc_prior, log_var_p=2. * xi_disc_prior,
                                                        axis=(1, 3))
        return kl_norm, kl_dis

    def load_prior_values(self):
        shape = (self.batch_size, self.nets.disc_latent_in, self.sample_size, self.nets.disc_var_num)
        self.mu_0, self.xi_0 = initialize_mu_and_xi_for_logistic(shape=shape)


class OptIGRDis(OptIGR):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.use_continuous = False

    def reparameterize(self, params_broad):
        mu, xi = params_broad
        self.load_prior_values()
        self.select_distribution(mu, xi)
        self.dist.generate_sample()
        z_discrete = [self.dist.psi]
        return z_discrete


class OptPlanarNF(OptIGR):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)

    def select_distribution(self, mu, xi):
        self.dist = IGR_Planar(mu=mu, xi=xi, planar_flow=self.nets.planar_flow,
                               temp=self.temp, sample_size=self.sample_size)


class OptPlanarNFDis(OptIGRDis):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)

    def select_distribution(self, mu, xi):
        self.dist = IGR_Planar(mu=mu, xi=xi, planar_flow=self.nets.planar_flow,
                               temp=self.temp, sample_size=self.sample_size)


class OptSBFinite(OptIGR):

    def __init__(self, nets, optimizer, hyper, use_continuous):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.prior_file = hyper['prior_file']
        self.use_continuous = use_continuous

    def reparameterize(self, params_broad):
        z = []
        self.load_prior_values()
        if self.use_continuous:
            mean, log_var, mu, xi = params_broad
            z_norm = sample_normal(mean=mean, log_var=log_var)
            z.append(z_norm)
        else:
            mu, xi = params_broad
        self.select_distribution(mu, xi)
        self.dist.generate_sample()
        self.n_required = self.dist.psi.shape[1]
        z_discrete = self.complete_discrete_vector()

        z.append(z_discrete)
        return z

    def select_distribution(self, mu, xi):
        self.dist = IGR_SB_Finite(mu, xi, self.temp, self.sample_size)

    def complete_discrete_vector(self):
        z_discrete = self.dist.psi
        return z_discrete

    def load_prior_values(self):
        with open(file=self.prior_file, mode='rb') as f:
            parameters = pickle.load(f)

        mu_0 = tf.constant(parameters['mu'], dtype=tf.float32)
        xi_0 = tf.constant(parameters['xi'], dtype=tf.float32)
        categories_n = mu_0.shape[1]

        self.mu_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=mu_0, batch_size=self.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.nets.disc_var_num)
        self.xi_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=xi_0, batch_size=self.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.nets.disc_var_num)


class OptSB(OptSBFinite):

    def __init__(self, nets, optimizer, hyper, use_continuous):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper, use_continuous=use_continuous)
        self.max_categories = hyper['latent_discrete_n']
        self.threshold = hyper['threshold']
        self.truncation_option = hyper['truncation_option']
        self.prior_file = hyper['prior_file']
        self.quantile = 50
        self.use_continuous = use_continuous

    def select_distribution(self, mu, xi):
        self.dist = IGR_SB(mu, xi, sample_size=self.sample_size, temp=self.temp, threshold=self.threshold)
        self.dist.truncation_option = self.truncation_option
        self.dist.quantile = self.quantile

    def complete_discrete_vector(self):
        batch_size, n_required = self.dist.psi.shape[0], self.dist.psi.shape[1]
        missing = self.max_categories - n_required
        zeros = tf.constant(value=0., dtype=tf.float32,
                            shape=(batch_size, missing, self.sample_size, self.num_of_vars))
        z_discrete = tf.concat([self.dist.psi, zeros], axis=1)
        return z_discrete


# ===========================================================================================================
def compute_loss(log_px_z, kl_norm, kl_dis, run_jv=False,
                 gamma=tf.constant(1.), discrete_c=tf.constant(0.), continuous_c=tf.constant(0.)):
    if run_jv:
        loss = -tf.reduce_mean(log_px_z - gamma * tf.math.abs(kl_norm - continuous_c)
                               - gamma * tf.math.abs(kl_dis - discrete_c))
    else:
        kl = kl_norm + kl_dis
        elbo = tf.reduce_mean(log_px_z - kl)
        loss = -elbo
    return loss


def compute_log_bernoulli_pdf(x, x_logit):
    x_broad = infer_shape_from(v=x_logit, x=x)
    cross_ent = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_broad, logits=x_logit)
    log_px_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return log_px_z


def compute_log_gaussian_pdf(x, x_logit):
    mu, xi = x_logit
    mu = tf.math.sigmoid(mu)
    xi = 1.e-6 + tf.math.softplus(xi)
    pi = 3.141592653589793

    x_broad = infer_shape_from(v=mu, x=x)

    log_pixel = - 0.5 * ((x_broad - mu) / xi) ** 2. - 0.5 * tf.math.log(2 * pi) - tf.math.log(1.e-8 + xi)
    log_px_z = tf.reduce_sum(log_pixel, axis=[1, 2, 3])
    return log_px_z


def infer_shape_from(v, x):
    batch_size, image_size, sample_size = v.shape[0], v.numpy().shape[1:4], v.shape[4]
    x_w_extra_col = tf.reshape(x, shape=(batch_size,) + image_size + (1,))
    x_broad = tf.broadcast_to(x_w_extra_col, shape=(batch_size,) + image_size + (sample_size,))
    return x_broad


def sample_normal(mean, log_var):
    epsilon = tf.random.normal(shape=mean.shape)
    z_norm = mean + tf.math.exp(log_var * 0.5) * epsilon
    return z_norm


def sample_kl_norm(z_norm, mean, log_var):
    log_pz = compute_log_normal_pdf(sample=z_norm, mean=0., log_var=0.)
    log_qz_x = compute_log_normal_pdf(sample=z_norm, mean=mean, log_var=log_var)
    kl_norm = log_qz_x - log_pz
    return kl_norm


def calculate_simple_closed_gauss_kl(mean, log_var):
    kl_norm = 0.5 * tf.reduce_sum(tf.math.exp(log_var) + tf.math.pow(mean, 2) - log_var - tf.constant(1.),
                                  axis=1)
    return kl_norm


def calculate_general_closed_form_gauss_kl(mean_q, log_var_q, mean_p, log_var_p, axis=(1,)):
    var_q = tf.math.exp(log_var_q)
    var_p = tf.math.exp(log_var_p)

    trace_term = tf.reduce_sum(var_q / var_p - 1., axis=axis)
    means_term = tf.reduce_sum(tf.math.pow(mean_q - mean_p, 2) / var_p, axis=axis)
    log_det_term = tf.reduce_sum(log_var_p - log_var_q, axis=axis)
    kl_norm = 0.5 * (trace_term + means_term + log_det_term)
    return kl_norm


def compute_log_normal_pdf(sample, mean, log_var):
    pi = 3.141592653589793
    log2pi = -0.5 * tf.math.log(2 * pi)
    log_exp_sum = -0.5 * (sample - mean) ** 2 * tf.math.exp(-log_var)
    log_normal_pdf = tf.reduce_sum(log2pi + -0.5 * log_var + log_exp_sum, axis=1)
    return log_normal_pdf


def calculate_categorical_closed_kl(log_alpha):
    offset = 1.e-20
    categories_n = tf.constant(log_alpha.shape[1], dtype=tf.float32)
    log_uniform_inv = tf.math.log(categories_n)
    pi = tf.math.softmax(log_alpha, axis=1)
    kl_discrete = tf.reduce_sum(pi * tf.math.log(pi + offset), axis=1) + log_uniform_inv
    return kl_discrete


def sample_kl_exp_gs(log_psi, log_pi, temp):
    uniform_probs = get_broadcasted_uniform_probs(shape=log_psi.shape)
    log_pz = compute_log_exp_gs_dist(log_psi=log_psi, logits=tf.math.log(uniform_probs), temp=temp)
    log_qz_x = compute_log_exp_gs_dist(log_psi=log_psi, logits=log_pi, temp=temp)
    kl_discrete = tf.math.reduce_sum(log_qz_x - log_pz, axis=2)
    return kl_discrete


def get_broadcasted_uniform_probs(shape):
    batch_n, categories_n, sample_size, disc_var_num = shape
    uniform_probs = tf.constant([1 / categories_n for _ in range(categories_n)], dtype=tf.float32,
                                shape=(1, categories_n, 1, 1))
    uniform_probs = shape_prior_to_sample_size_and_discrete_var_num(uniform_probs, batch_n,
                                                                    categories_n, sample_size, disc_var_num)
    return uniform_probs


def shape_prior_to_sample_size_and_discrete_var_num(prior_param, batch_size, categories_n,
                                                    sample_size, discrete_var_num):
    prior_param = tf.reshape(prior_param, shape=(1, categories_n, 1, 1))
    prior_param = tf.broadcast_to(prior_param, shape=(batch_size, categories_n, 1, 1))
    prior_param = tf.broadcast_to(prior_param, shape=(batch_size, categories_n, sample_size, 1))
    prior_param = tf.broadcast_to(prior_param,
                                  shape=(batch_size, categories_n, sample_size, discrete_var_num))
    return prior_param


def reshape_and_stack_z(z):
    if len(z) > 1:
        z = tf.concat(z, axis=1)
        z = flatten_discrete_variables(original_z=z)
    else:
        z = flatten_discrete_variables(original_z=z[0])
    return z


def flatten_discrete_variables(original_z):
    batch_n, disc_latent_n, sample_size, disc_var_num = original_z.shape
    z_discrete = tf.reshape(original_z, shape=(batch_n, disc_var_num * disc_latent_n, sample_size))
    return z_discrete

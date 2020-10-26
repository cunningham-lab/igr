import pickle
import tensorflow as tf
from os import environ as os_env
from tensorflow_probability import distributions as tfpd
from Utils.Distributions import IGR_I, IGR_Planar, IGR_SB, IGR_SB_Finite
from Utils.Distributions import GS, compute_log_exp_gs_dist
from Utils.Distributions import project_to_vertices_via_softmax_pp
from Utils.Distributions import project_to_vertices
from Utils.Distributions import compute_igr_log_probs
from Utils.Distributions import compute_igr_probs
from Utils.Distributions import iterative_sb
from Models.VAENet import RelaxCovNet
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'


class OptVAE:

    def __init__(self, nets, optimizer, hyper):
        self.nets = nets
        self.optimizer = optimizer
        self.dtype = hyper['dtype']
        self.batch_size = hyper['batch_n']
        self.n_required = hyper['n_required']
        self.sample_size = hyper['sample_size']
        self.sample_size_training = hyper['sample_size']
        self.sample_size_testing = hyper['sample_size_testing']
        self.num_of_vars = hyper['num_of_discrete_var']
        self.dataset_name = hyper['dataset_name']
        self.model_type = hyper['model_type']
        self.test_with_one_hot = hyper['test_with_one_hot']
        self.sample_from_cont_kl = hyper['sample_from_cont_kl']
        self.sample_from_disc_kl = hyper['sample_from_disc_kl']
        self.temp = tf.constant(value=hyper['temp'], dtype=self.dtype)
        self.stick_the_landing = hyper['stick_the_landing']
        self.iter_count = 0
        self.estimate_kl_w_n = self.sample_size
        self.run_iwae = False
        self.train_loss_mean = tf.keras.metrics.Mean()

    def perform_fwd_pass(self, x, test_with_one_hot=False):
        self.set_hyper_for_testing(test_with_one_hot)
        params = self.nets.encode(x, self.batch_size)
        z = self.reparameterize(params_broad=params)
        x_logit = self.decode_w_or_wo_one_hot(z, test_with_one_hot)
        return z, x_logit, params

    def set_hyper_for_testing(self, test_with_one_hot):
        if test_with_one_hot:
            self.sample_size = self.sample_size_testing
            self.estimate_kl_w_n = 1 * int(1.e3)
        else:
            self.sample_size = self.sample_size_training
            self.estimate_kl_w_n = self.sample_size_training

    def decode_w_or_wo_one_hot(self, z, test_with_one_hot):
        if test_with_one_hot:
            _, categories_n, _, _ = z[-1].shape
            zz = []
            for idx in range(len(z)):
                one_hot = project_to_vertices(z[idx], categories_n)
                one_hot = one_hot[:, :, :self.sample_size, :]
                zz.append(one_hot)
            x_logit = self.decode(z=zz)
        else:
            x_logit = self.decode(z=z)
        return x_logit

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
        batch_n, _ = z[0].shape[0], z[0].shape[2]
        z = reshape_and_stack_z(z=z)
        x_logit = tf.TensorArray(dtype=self.dtype, size=self.sample_size,
                                 element_shape=(batch_n,) + self.nets.image_shape)
        for i in tf.range(self.sample_size):
            x_logit = x_logit.write(index=i, value=self.nets.decode(z[:, :, i])[0])
        x_logit = tf.transpose(x_logit.stack(), perm=[1, 2, 3, 4, 0])
        return x_logit

    def decode_gaussian(self, z):
        z = reshape_and_stack_z(z=z)
        batch_n, _ = z.shape[0], z.shape[2]
        mu = tf.TensorArray(dtype=self.dtype, size=self.sample_size,
                            element_shape=(batch_n,) + self.nets.image_shape)
        xi = tf.TensorArray(dtype=self.dtype, size=self.sample_size,
                            element_shape=(batch_n,) + self.nets.image_shape)
        for i in tf.range(self.sample_size):
            z_mu, z_xi = self.nets.decode(z[:, :, i])
            mu = mu.write(index=i, value=z_mu)
            xi = xi.write(index=i, value=z_xi)
        mu = tf.transpose(mu.stack(), perm=[1, 2, 3, 4, 0])
        xi = tf.transpose(xi.stack(), perm=[1, 2, 3, 4, 0])
        x_logit = [mu, xi]
        return x_logit

    def compute_kl_elements(self, z, params_broad,
                            sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot):
        mean, log_var = params_broad
        if sample_from_cont_kl:
            kl_norm = sample_kl_norm(z_norm=z, mean=mean, log_var=log_var)
        else:
            kl_norm = calculate_simple_closed_gauss_kl(mean=mean, log_var=log_var)
        kl_dis = tf.constant(0.) if sample_from_disc_kl else tf.constant(0.)
        return kl_norm, kl_dis

    def compute_loss(self, x, x_logit, z, params_broad,
                     sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot,
                     run_iwae):
        if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
            log_px_z = compute_log_gaussian_pdf(
                x=x, x_logit=x_logit, sample_size=self.sample_size)
        else:
            log_px_z = compute_log_bernoulli_pdf(
                x=x, x_logit=x_logit, sample_size=self.sample_size)
        kl_norm, kl_dis = self.compute_kl_elements(z, params_broad,
                                                   sample_from_cont_kl,
                                                   sample_from_disc_kl,
                                                   test_with_one_hot)
        loss = compute_loss(log_px_z=log_px_z, kl=kl_norm + kl_dis,
                            sample_size=self.sample_size, run_iwae=run_iwae)
        return loss

    @tf.function()
    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl,
                                           sample_from_disc_kl):
        z, x_logit, params_broad = self.perform_fwd_pass(x, self.test_with_one_hot)
        loss = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                 sample_from_cont_kl=sample_from_cont_kl,
                                 sample_from_disc_kl=sample_from_disc_kl,
                                 test_with_one_hot=self.test_with_one_hot,
                                 run_iwae=self.run_iwae)
        return loss

    @tf.function()
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x,
                                                             test_with_one_hot=False)
            loss = self.compute_loss(x=x, x_logit=x_logit, z=z,
                                     params_broad=params_broad,
                                     sample_from_cont_kl=self.sample_from_cont_kl,
                                     sample_from_disc_kl=self.sample_from_disc_kl,
                                     test_with_one_hot=False,
                                     run_iwae=False)
        gradients = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nets.trainable_variables))

        return loss

    def train_on_epoch(self, train_dataset, take):
        self.train_loss_mean.reset_states()
        for x_train in train_dataset.take(take):
            self.perform_train_step(x_train)

    def perform_train_step(self, x_train):
        loss = self.compute_gradients(x=x_train)
        self.iter_count += 1
        self.train_loss_mean(loss)


class OptDLGMM(OptVAE):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.mu_prior = self.create_separated_prior_means()
        self.log_var_prior = tf.zeros_like(self.mu_prior)

    def create_separated_prior_means(self):
        mu_prior = tf.zeros(shape=(self.batch_size, 1,
                                   self.sample_size, self.num_of_vars))
        mult = 1. / tf.sqrt(tf.constant(self.num_of_vars, dtype=self.dtype))
        for _ in range(self.n_required - 1):
            mu = tf.ones(shape=(self.batch_size, 1,
                                self.sample_size, self.num_of_vars))
            u = tf.random.uniform(shape=mu.shape)
            mu = tf.where(u < 0.5, -1.0, 1.0)
            mu = mult * mu
            mu_prior = tf.concat([mu_prior, mu], axis=1)
        return mu_prior

    def reparameterize(self, params_broad):
        log_a, log_b, mean, log_var = params_broad
        a, b = tf.math.exp(log_a)[:, :, 0, :], tf.math.exp(log_b)[:, :, 0, :]
        kumar = tfpd.Kumaraswamy(concentration0=a, concentration1=b)
        z_kumar = kumar.sample(sample_shape=self.sample_size)
        z_kumar = tf.transpose(z_kumar, perm=[1, 2, 0, 3])
        z_kumar = tf.clip_by_value(z_kumar, 1.e-4, 0.9999)
        z_norm = sample_normal(mean=tf.repeat(mean, self.sample_size, axis=2),
                               log_var=tf.repeat(log_var, self.sample_size, axis=2))
        z = [z_kumar, z_norm]
        return z

    def decode(self, z):
        batch_n = z[-1].shape[0]
        n_required = self.n_required if self.n_required is not None else 1
        x_logit = tf.TensorArray(dtype=self.dtype, size=self.sample_size,
                                 element_shape=(batch_n,) + self.nets.image_shape +
                                 (n_required,))
        for i in range(self.sample_size):
            value = tf.expand_dims(self.nets.decode(z[-1][:, 0, i, :])[0], axis=-1)
            for j in range(1, n_required):
                logits = tf.expand_dims(self.nets.decode(z[-1][:, j, i, :])[0],
                                        axis=-1)
                value = tf.concat((value, logits), axis=4)
            x_logit = x_logit.write(index=i, value=value)
        x_logit = tf.transpose(x_logit.stack(), perm=[1, 2, 3, 4, 0, 5])
        return x_logit

    @tf.function()
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x,
                                                             test_with_one_hot=False)
            loss = self.compute_loss(x=x, x_logit=x_logit, z=z,
                                     params_broad=params_broad,
                                     sample_from_cont_kl=self.sample_from_cont_kl,
                                     sample_from_disc_kl=self.sample_from_disc_kl,
                                     test_with_one_hot=False,
                                     run_iwae=False)
        gradients = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        grads = []
        for grad in gradients:
            grad = tf.where(tf.math.is_nan(grad), 0., grad)
            grads.append(grad)
        gradients = grads
        self.optimizer.apply_gradients(zip(gradients, self.nets.trainable_variables))
        return loss

    @tf.function()
    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl,
                                           sample_from_disc_kl):
        z, x_logit, params_broad = self.perform_fwd_pass(x, self.test_with_one_hot)
        loss = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                 sample_from_cont_kl=sample_from_cont_kl,
                                 sample_from_disc_kl=sample_from_disc_kl,
                                 test_with_one_hot=self.test_with_one_hot,
                                 run_iwae=self.run_iwae)
        return loss

    def compute_loss(self, x, x_logit, z, params_broad,
                     sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot,
                     run_iwae):
        log_a, log_b, mean, log_var = params_broad
        z_kumar, z_norm = z
        pi = iterative_sb(z_kumar)
        pi = tf.clip_by_value(pi, 1.e-4, 0.9999)

        log_px_z = self.compute_log_px_z(x, x_logit, pi)
        log_pz = self.compute_log_pz(z_norm, pi)
        log_a = tf.stop_gradient(log_a)
        log_b = tf.stop_gradient(log_b)
        mean = tf.stop_gradient(mean)
        log_var = tf.stop_gradient(log_var)
        log_qpi_x = self.compute_log_qpi_x(z_kumar, log_a, log_b)
        log_qz_x = self.compute_log_qz_x(z_norm, pi, mean, log_var)
        loss = log_qz_x + log_qpi_x - log_px_z - log_pz
        return loss

    def compute_log_px_z(self, x, x_logit, pi):
        n_required = self.n_required if self.n_required is not None else 1
        pi_bar_k = tf.reduce_mean(pi, axis=(2, 3))
        x_broad = tf.repeat(tf.expand_dims(x, -1), axis=-1,
                            repeats=self.sample_size)
        x_broad = tf.repeat(tf.expand_dims(x_broad, -1), axis=-1,
                            repeats=n_required)
        cross_ent = - tf.nn.sigmoid_cross_entropy_with_logits(labels=x_broad,
                                                              logits=x_logit)
        log_px_z = tf.reduce_sum(cross_ent, axis=(1, 2, 3))
        log_px_z = tf.reduce_mean(log_px_z, axis=1)
        log_px_z = tf.reduce_sum(pi_bar_k * log_px_z, axis=-1)
        log_px_z = tf.reduce_mean(log_px_z)
        return log_px_z

    def compute_log_pz(self, z, pi):
        sample_axis = 2
        mu_prior = self.mu_prior[:, :self.n_required, :, :]
        log_var_prior = self.log_var_prior[:, :self.n_required, :, :]
        pi_bar_k = tf.reduce_mean(pi, axis=sample_axis, keepdims=True)
        pi_const = tf.constant(3.141592653589793, dtype=self.mu_prior.dtype)
        log2pi = -0.5 * tf.math.log(2 * pi_const)
        log_exp_sum = (-0.5 * (z - mu_prior) ** 2 * tf.math.exp(-log_var_prior))
        log_pz = tf.reduce_mean(log2pi + -0.5 * log_var_prior + log_exp_sum,
                                axis=sample_axis, keepdims=True)
        log_pz = tf.reduce_sum(log_pz, axis=3, keepdims=True)
        log_pz = tf.reduce_sum(pi_bar_k * log_pz, axis=(1, 2, 3))
        log_pz = tf.reduce_mean(log_pz)
        return log_pz

    def compute_log_qpi_x(self, z_kumar, log_a, log_b):
        a, b = tf.math.exp(log_a), tf.math.exp(log_b)
        kumar = tfpd.Kumaraswamy(concentration0=a, concentration1=b)
        log_qpi_x = kumar.log_prob(z_kumar)
        log_qpi_x = tf.reduce_mean(log_qpi_x, axis=2, keepdims=True)
        log_qpi_x = tf.reduce_sum(log_qpi_x, axis=(1, 2, 3))
        log_qpi_x = tf.reduce_mean(log_qpi_x, axis=0)
        return log_qpi_x

    def compute_kld(self, log_a, log_b):
        sample_axis = 2
        a, b = tf.math.exp(log_a), tf.math.exp(log_b)
        harmonic_b = tf.math.digamma(b + 1.) - tf.math.digamma(tf.ones_like(b))
        log_qpi_x = (1. / b - 1.) + (1. / a - 1.) * harmonic_b + log_a + log_b
        log_qpi_x = tf.reduce_mean(log_qpi_x, axis=sample_axis, keepdims=True)
        log_qpi_x = tf.reduce_sum(log_qpi_x, axis=(1, 2, 3))
        log_qpi_x = tf.reduce_mean(log_qpi_x, axis=0)
        return log_qpi_x

    def compute_log_qz_x(self, z, pi, mean, log_var):
        cat_axis, sample_axis, dim_axis = 1, 2, 3
        log_pi = tf.math.log(pi)
        pi_const = tf.constant(3.141592653589793, dtype=mean.dtype)
        log2pi = -0.5 * tf.math.log(2 * pi_const)
        log_exp_sum = (-0.5 * (z - mean) ** 2 * tf.math.exp(-log_var))
        lse = tf.reduce_sum(log2pi + -0.5 * log_var + log_exp_sum,
                            axis=dim_axis, keepdims=True)
        lse += log_pi
        log_qz_x = tf.math.reduce_logsumexp(lse, axis=(cat_axis, dim_axis),
                                            keepdims=True)
        log_qz_x = tf.reduce_mean(log_qz_x, axis=sample_axis)
        log_qz_x = tf.reduce_sum(log_qz_x, axis=cat_axis)
        log_qz_x = tf.reduce_mean(log_qz_x)
        return log_qz_x


class OptDLGMM_Var(OptDLGMM):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.mu_prior = self.create_separated_prior_means()
        self.log_var_prior = tf.zeros_like(self.mu_prior)
        self.threshold = 0.9999
        self.quantile = 75

    def reparameterize(self, params_broad):
        log_a, log_b, mean, log_var = params_broad
        a, b = tf.math.exp(log_a)[:, :, 0, :], tf.math.exp(log_b)[:, :, 0, :]
        kumar = tfpd.Kumaraswamy(concentration0=a, concentration1=b)
        z_kumar = kumar.sample(sample_shape=self.sample_size)
        z_kumar = tf.transpose(z_kumar, perm=[1, 2, 0, 3])
        z_kumar = tf.clip_by_value(z_kumar, 1.e-4, 0.9999)
        z_norm = sample_normal(mean=tf.repeat(mean, self.sample_size, axis=2),
                               log_var=tf.repeat(log_var, self.sample_size, axis=2))
        z = [z_kumar, z_norm]
        self.aux = IGR_SB(log_a, log_b, self.temp, self.sample_size,
                          threshold=self.threshold)
        self.aux.kappa = z_kumar
        self.pi = self.aux.transform()
        self.n_required = self.pi.shape[1]
        return z

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x,
                                                             test_with_one_hot=False)
            loss = self.compute_loss(x=x, x_logit=x_logit, z=z,
                                     params_broad=params_broad,
                                     sample_from_cont_kl=self.sample_from_cont_kl,
                                     sample_from_disc_kl=self.sample_from_disc_kl,
                                     test_with_one_hot=False,
                                     run_iwae=False)
        gradients = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nets.trainable_variables))

        return loss

    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl,
                                           sample_from_disc_kl):
        z, x_logit, params_broad = self.perform_fwd_pass(x, self.test_with_one_hot)
        loss = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                 sample_from_cont_kl=sample_from_cont_kl,
                                 sample_from_disc_kl=sample_from_disc_kl,
                                 test_with_one_hot=self.test_with_one_hot,
                                 run_iwae=self.run_iwae)
        return loss

    def compute_loss(self, x, x_logit, z, params_broad,
                     sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot,
                     run_iwae):
        log_a, log_b, mean, log_var = params_broad
        z_kumar, z_norm = z
        pi = self.pi
        output = self.threshold_params(params_broad, z, pi)
        log_a, log_b, mean, log_var, z_kumar, z_norm, pi = output

        log_px_z = self.compute_log_px_z(x, x_logit, pi)
        log_pz = self.compute_log_pz(z_norm, pi)
        log_a = tf.stop_gradient(log_a)
        log_b = tf.stop_gradient(log_b)
        mean = tf.stop_gradient(mean)
        log_var = tf.stop_gradient(log_var)
        log_qpi_x = self.compute_log_qpi_x(z_kumar, log_a, log_b)
        log_qz_x = self.compute_log_qz_x(z_norm, pi, mean, log_var)
        loss = log_qz_x + log_qpi_x - log_px_z - log_pz
        return loss

    def threshold_params(self, params_broad, z, pi):
        new_params_broad = []
        all_params = [params_broad, z, [pi]]
        for param_list in all_params:
            for param in param_list:
                param = param[:, :self.n_required, :, :]
                new_params_broad.append(param)
        return new_params_broad


class OptDLGMMIGR(OptDLGMM):
    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.mu_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.xi_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.dist = tfpd.LogitNormal(self.mu_0, tf.math.exp(self.xi_0))

    def reparameterize(self, params_broad):
        mu, xi, mean, log_var = params_broad
        self.dist = tfpd.LogitNormal(loc=mu[:, :, 0, :],
                                     scale=tf.math.exp(xi)[:, :, 0, :])
        z_partition = self.dist.sample(sample_shape=self.sample_size)
        z_partition = tf.transpose(z_partition, perm=[1, 2, 0, 3])
        z_norm = sample_normal(mean=tf.repeat(mean, self.sample_size, axis=2),
                               log_var=tf.repeat(log_var, self.sample_size, axis=2))
        z = [z_partition, z_norm]
        return z

    @tf.function()
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x,
                                                             test_with_one_hot=False)
            loss = self.compute_loss(x=x, x_logit=x_logit, z=z,
                                     params_broad=params_broad,
                                     sample_from_cont_kl=self.sample_from_cont_kl,
                                     sample_from_disc_kl=self.sample_from_disc_kl,
                                     test_with_one_hot=False,
                                     run_iwae=False)
        gradients = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nets.trainable_variables))
        return loss

    def compute_loss(self, x, x_logit, z, params_broad,
                     sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot,
                     run_iwae):
        mu, xi, mean, log_var = params_broad
        z_partition, z_norm = z
        pi = iterative_sb(z_partition)
        pi = tf.clip_by_value(pi, 1.e-4, 0.9999)
        z_partition = tf.clip_by_value(z_partition, 1.e-4, 0.9999)

        log_px_z = self.compute_log_px_z(x, x_logit, pi)
        log_pz = self.compute_log_pz(z_norm, pi)
        mu = tf.stop_gradient(mu)
        xi = tf.stop_gradient(xi)
        mean = tf.stop_gradient(mean)
        log_var = tf.stop_gradient(log_var)
        log_qpi_x = self.compute_kld(z_partition, mu, xi)
        log_qz_x = self.compute_log_qz_x(z_norm, pi, mean, log_var)
        loss = log_qz_x + log_qpi_x - log_px_z - log_pz
        return loss

    def compute_kld(self, z_partition, mu, xi):
        self.dist = tfpd.LogitNormal(loc=mu, scale=tf.math.exp(xi))
        log_qpi_x = self.dist.log_prob(z_partition)
        log_qpi_x = tf.reduce_mean(log_qpi_x, axis=2, keepdims=True)
        log_qpi_x = tf.reduce_sum(log_qpi_x, axis=(1, 2, 3))
        log_qpi_x = tf.reduce_mean(log_qpi_x, axis=0)
        return log_qpi_x


class OptDLGMMIGR_SB(OptDLGMMIGR):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.max_categories = hyper['latent_discrete_n']
        self.truncation_option = 'quantile'
        self.threshold = 0.9999
        self.quantile = 75

    def reparameterize(self, params_broad):
        mu, xi, mean, log_var = params_broad
        self.aux = IGR_SB(mu, xi, self.temp, self.sample_size,
                          threshold=self.threshold)
        self.aux.generate_sample()
        z_partition = tf.math.sigmoid(self.aux.kappa)
        z_norm = sample_normal(mean=tf.repeat(mean, self.sample_size, axis=2),
                               log_var=tf.repeat(log_var, self.sample_size, axis=2))
        z = [z_partition, z_norm]
        self.pi = self.aux.transform()
        self.n_required = self.pi.shape[1]
        return z

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x,
                                                             test_with_one_hot=False)
            loss = self.compute_loss(x=x, x_logit=x_logit, z=z,
                                     params_broad=params_broad,
                                     sample_from_cont_kl=self.sample_from_cont_kl,
                                     sample_from_disc_kl=self.sample_from_disc_kl,
                                     test_with_one_hot=False,
                                     run_iwae=False)
        gradients = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nets.trainable_variables))

        return loss

    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl,
                                           sample_from_disc_kl):
        z, x_logit, params_broad = self.perform_fwd_pass(x, self.test_with_one_hot)
        loss = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                 sample_from_cont_kl=sample_from_cont_kl,
                                 sample_from_disc_kl=sample_from_disc_kl,
                                 test_with_one_hot=self.test_with_one_hot,
                                 run_iwae=self.run_iwae)
        return loss

    def compute_loss(self, x, x_logit, z, params_broad,
                     sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot,
                     run_iwae):
        pi = self.pi
        pi = tf.clip_by_value(pi, 1.e-4, 0.9999)
        output = self.threshold_params(params_broad, z, pi)
        mu, xi, mean, log_var, z_partition, z_norm, pi = output
        self.dist = tfpd.LogitNormal(loc=mu, scale=tf.math.exp(xi))
        z_partition = tf.clip_by_value(z_partition, 1.e-4, 0.9999)

        log_px_z = self.compute_log_px_z(x, x_logit, pi)
        log_pz = self.compute_log_pz(z_norm, pi)
        log_qpi_x = self.compute_kld(z_partition, mu, xi)
        log_qz_x = self.compute_log_qz_x(z_norm, pi, mean, log_var)
        loss = log_qz_x + log_qpi_x - log_px_z - log_pz
        return loss

    def threshold_params(self, params_broad, z, pi):
        new_params_broad = []
        all_params = [params_broad, z, [pi]]
        for param_list in all_params:
            for param in param_list:
                param = param[:, :self.n_required, :, :]
                new_params_broad.append(param)
        return new_params_broad


class OptExpGSDis(OptVAE):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.dist = GS(log_pi=tf.constant(1., dtype=self.dtype, shape=(1, 1, 1, 1)),
                       temp=self.temp)
        self.log_psi = tf.constant(1., dtype=self.dtype, shape=(1, 1, 1, 1))

    def reparameterize(self, params_broad):
        self.dist = GS(log_pi=params_broad[0],
                       sample_size=self.sample_size, temp=self.temp)
        self.dist.generate_sample()
        self.n_required = self.dist.psi.shape[1]
        self.log_psi = self.dist.log_psi
        z_discrete = [self.dist.psi]
        return z_discrete

    def compute_kl_elements(self, z, params_broad, sample_from_cont_kl,
                            sample_from_disc_kl, test_with_one_hot):
        log_alpha = params_broad[0]
        if self.stick_the_landing:
            log_alpha = tf.stop_gradient(log_alpha)
        kl_norm = 0.
        kl_dis = self.compute_discrete_kl(log_alpha, sample_from_disc_kl)
        return kl_norm, kl_dis

    def compute_discrete_kl(self, log_alpha, sample_from_disc_kl):
        if sample_from_disc_kl:
            kl_dis = sample_kl_exp_gs(log_psi=self.log_psi, log_pi=log_alpha,
                                      temp=self.temp)
        else:
            kl_dis = calculate_categorical_closed_kl(log_alpha=log_alpha, normalize=True)
        return kl_dis


class OptRELAX(OptVAE):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizer=optimizers[0], hyper=hyper)
        self.optimizer_encoder = self.optimizer
        self.optimizer_decoder = optimizers[1]
        self.optimizer_var = optimizers[2]
        cov_net_shape = (self.n_required, self.sample_size, self.num_of_vars)
        self.relax_cov = RelaxCovNet(cov_net_shape, self.dtype)
        num_latents = self.n_required * self.num_of_vars
        shape = (1, self.n_required, self.sample_size, self.num_of_vars)
        initial_log_temp = tf.constant([1.6093 for _ in range(num_latents)],
                                       shape=(1, self.n_required, 1, self.num_of_vars))
        initial_log_temp = tf.broadcast_to(initial_log_temp, shape=shape)
        self.log_temp = tf.Variable(initial_log_temp, name='log_temp', trainable=True)
        self.decoder_vars = [
            v for v in self.nets.trainable_variables if 'decoder' in v.name]
        self.encoder_vars = [
            v for v in self.nets.trainable_variables if 'encoder' in v.name]
        self.con_net_vars = self.relax_cov.net.trainable_variables + [self.log_temp]
        self.iter_count = 0

    def compute_loss(self, z, x, params,
                     sample_from_cont_kl=None, sample_from_disc_kl=None,
                     test_with_one_hot=False, run_iwae=False):
        categories_n = tf.cast(z.shape[1], dtype=tf.float32)
        num_of_vars = tf.cast(z.shape[-1], dtype=tf.float32)

        x_logit = self.decode([z])
        log_px_z = compute_log_bernoulli_pdf(x=x, x_logit=x_logit,
                                             sample_size=self.sample_size)
        log_p = - num_of_vars * tf.math.log(categories_n)
        log_qz_x = self.compute_log_pmf(z=z, params=params)
        kl = log_p - tf.reduce_mean(log_qz_x)
        recon = -tf.math.reduce_mean(log_px_z)
        loss = recon - kl
        return loss

    def offload_params(self, params):
        self.log_alpha = params[0]

    def compute_log_pmf(self, z, params):
        log_pmf = compute_log_categorical_pmf(z, params[0])
        return log_pmf

    def compute_log_pmf_grad(self, z):
        grad = z - tf.math.softmax(self.log_alpha, axis=1)
        return grad

    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl,
                                           sample_from_disc_kl):
        params = self.nets.encode(x, self.batch_size)
        self.offload_params(params)
        one_hot = self.get_relax_variables_from_params(x, params)[-1]
        loss = self.compute_loss(z=one_hot, x=x, params=params)
        return loss

    def compute_c_phi(self, z, x, params):
        r = tf.math.reduce_mean(self.relax_cov.net(z))
        c_phi = self.compute_loss(x=x, z=z, params=params) + r
        return c_phi

    @tf.function()
    def compute_gradients(self, x):
        with tf.GradientTape() as tape_cov:
            with tf.GradientTape(persistent=True) as tape:
                params = self.nets.encode(x, self.batch_size)
                self.offload_params(params)
                tape.watch(params)
                c_phi, c_phi_tilde, one_hot = self.get_relax_variables_from_params(
                    x, params)
                loss = self.compute_loss(x=x, z=one_hot, params=params)
                c_diff = tf.reduce_mean(c_phi - c_phi_tilde)
                log_qz_x_grad = tf.reduce_mean(self.compute_log_pmf_grad(z=one_hot),
                                               axis=2, keepdims=True)
                log_pmf = self.compute_log_pmf(one_hot, params)
            c_diff_grad = tape.gradient(target=c_diff, sources=params)
            relax_grad = self.compute_relax_grad(
                loss, c_phi_tilde, [log_qz_x_grad], c_diff_grad)

            c_diff_g = tape.gradient(target=c_diff, sources=self.encoder_vars)
            log_qz_x_g = tape.gradient(target=log_pmf, sources=self.encoder_vars)
            encoder_grads = self.compute_relax_grad(
                loss, c_phi_tilde, log_qz_x_g, c_diff_g)
            decoder_grads = tape.gradient(target=loss, sources=self.decoder_vars)

            variance = compute_grad_var_over_batch(relax_grad)
        cov_net_grad = tape_cov.gradient(target=variance, sources=self.con_net_vars)

        gradients = (encoder_grads, decoder_grads, cov_net_grad)
        self.apply_gradients(gradients)
        return loss

    def compute_relax_grad(self, loss, c_phi_tilde, log_qz_x_grad, c_phi_diff_grad):
        relax_grads = []
        diff = loss - c_phi_tilde
        for i in range(len(log_qz_x_grad)):
            relax = diff * log_qz_x_grad[i]
            relax += c_phi_diff_grad[i]
            relax += log_qz_x_grad[i]
            relax_grads.append(relax)
        return relax_grads

    def apply_gradients(self, gradients):
        encoder_grads, decoder_grads, cov_net_grad = gradients
        self.optimizer_encoder.apply_gradients(zip(encoder_grads, self.encoder_vars))
        self.optimizer_decoder.apply_gradients(zip(decoder_grads, self.decoder_vars))
        self.optimizer_var.apply_gradients(zip(cov_net_grad, self.con_net_vars))


class OptRELAXIGR(OptRELAX):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizers=optimizers, hyper=hyper)
        num_latents = self.n_required * self.num_of_vars
        shape = (1, self.n_required, self.sample_size, self.num_of_vars)
        initial_log_temp = tf.constant([-1.8972 for _ in range(num_latents)],
                                       shape=(1, self.n_required, 1, self.num_of_vars),
                                       dtype=self.dtype)
        initial_log_temp = tf.broadcast_to(initial_log_temp, shape=shape)
        self.log_temp = tf.Variable(initial_log_temp, name='log_temp', trainable=True)
        cov_net_shape = (self.n_required + 1, self.sample_size, self.num_of_vars)
        self.relax_cov = RelaxCovNet(cov_net_shape, self.dtype)
        self.con_net_vars = self.relax_cov.net.trainable_variables + [self.log_temp]

    def compute_loss(self, z, x, params,
                     sample_from_cont_kl=None, sample_from_disc_kl=None,
                     test_with_one_hot=False):
        categories_n = tf.cast(z.shape[1], dtype=self.dtype)
        num_of_vars = tf.cast(z.shape[-1], dtype=self.dtype)

        x_logit = self.decode([z])
        log_px_z = compute_log_bernoulli_pdf(
            x=x, x_logit=x_logit, sample_size=self.sample_size)
        log_p = - num_of_vars * tf.math.log(categories_n)
        log_p_discrete = compute_igr_log_probs(self.mu, self.sigma)
        p_discrete = compute_igr_probs(self.mu, self.sigma)
        kl = tf.math.reduce_mean(tf.math.reduce_sum(
            log_p_discrete * p_discrete, axis=(1, 3)))
        kl -= log_p
        recon = -tf.math.reduce_mean(log_px_z)
        loss = recon - kl
        return loss

    def offload_params(self, params):
        mu, xi = params
        # Transformations to ensure numerical stability of the integral
        self.mu = tf.constant(1., self.dtype) * tf.math.tanh(mu)
        self.sigma = (tf.constant(0.5, self.dtype) * tf.math.sigmoid(xi) +
                      tf.constant(0.5, self.dtype))

    def transform_params_into_log_probs(self, params):
        log_probs = compute_igr_log_probs(self.mu, self.sigma)
        return log_probs

    def compute_log_pmf(self, z, params):
        log_probs = self.transform_params_into_log_probs(params)
        log_categorical_pmf = tf.math.reduce_sum(z * log_probs, axis=1)
        log_categorical_pmf = tf.math.reduce_sum(log_categorical_pmf, axis=(1, 2))
        return log_categorical_pmf

    def get_relax_variables_from_params(self, x, params):
        z_un = (self.mu +
                self.sigma * tf.random.normal(shape=self.mu.shape, dtype=self.dtype))
        z = project_to_vertices_via_softmax_pp(z_un / tf.math.exp(self.log_temp))
        z_un1 = (self.mu +
                 self.sigma * tf.random.normal(shape=self.mu.shape, dtype=self.dtype))
        z1 = project_to_vertices_via_softmax_pp(z_un1 / tf.math.exp(self.log_temp))
        one_hot = project_to_vertices(z, categories_n=self.n_required + 1)
        c_phi = self.compute_c_phi(z=z1, x=x, params=params)
        return c_phi, z_un, one_hot

    @tf.function()
    def compute_gradients(self, x):
        with tf.GradientTape() as tape_cov:
            with tf.GradientTape(persistent=True) as tape:
                params = self.nets.encode(x, self.batch_size)
                self.offload_params(params)
                tape.watch(params)
                c_phi, z_un, one_hot = self.get_relax_variables_from_params(x, params)
                loss = self.compute_loss(x=x, params=params, z=one_hot)
                log_pmf = self.compute_log_pmf(z=one_hot, params=params)

            c_phi_g = tape.gradient(target=c_phi, sources=params)
            log_cat_g = tape.gradient(target=log_pmf, sources=params)
            relax_grad = self.compute_relax_grad(loss, c_phi, log_cat_g, c_phi_g)

            c_phi_grad = tape.gradient(target=c_phi, sources=self.encoder_vars)
            log_qz_x_grad = tape.gradient(target=log_pmf, sources=self.encoder_vars)
            encoder_grads = self.compute_relax_grad(
                loss, c_phi, log_qz_x_grad, c_phi_grad)
            decoder_grads = tape.gradient(target=loss, sources=self.decoder_vars)

            variance = compute_grad_var_over_batch(relax_grad[0])
        cov_net_grad = tape_cov.gradient(target=variance, sources=self.con_net_vars)

        gradients = (encoder_grads, decoder_grads, cov_net_grad)
        self.apply_gradients(gradients)
        return loss


class OptRELAXGSDis(OptRELAX):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizers=optimizers, hyper=hyper)

    def get_relax_variables_from_params(self, x, params):
        u = tf.random.uniform(shape=(self.batch_size, self.n_required,
                                     self.sample_size, self.num_of_vars))
        z_un = self.log_alpha - tf.math.log(-tf.math.log(u + 1.e-20) + 1.e-20)
        one_hot = project_to_vertices(z_un, categories_n=self.n_required)
        z_tilde_un = sample_z_tilde_cat(one_hot, self.log_alpha)

        z = tf.math.softmax(z_un / tf.math.exp(self.log_temp) + self.log_alpha, axis=1)
        z_tilde = tf.math.softmax(z_tilde_un / tf.math.exp(self.log_temp) +
                                  self.log_alpha, axis=1)

        c_phi = self.compute_c_phi(z=z, x=x, params=params)
        c_phi_tilde = self.compute_c_phi(z=z_tilde, x=x, params=params)
        return c_phi, c_phi_tilde, one_hot


class OptRELAXBerDis(OptRELAXGSDis):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizers=optimizers, hyper=hyper)

    def compute_log_pmf(self, z, log_probs):
        log_pmf = bernoulli_loglikelihood(b=z, log_alpha=log_probs)
        log_pmf = tf.math.reduce_sum(log_pmf, axis=(1, 2, 3))
        return log_pmf

    def compute_log_pmf_grad(self, z, params):
        log_alpha = self.transform_params_into_log_probs(params)
        grad = bernoulli_loglikelihood_grad(z, log_alpha)
        return grad

    def get_relax_variables_from_params(self, x, params):
        log_alpha = params[0]
        u = tf.random.uniform(shape=log_alpha.shape)
        z_un = log_alpha + safe_log_prob(u) - safe_log_prob(1 - u)
        one_hot = tf.cast(tf.stop_gradient(z_un > 0), dtype=tf.float32)
        z_tilde_un = sample_z_tilde_ber(log_alpha=log_alpha, u=u)

        z = tf.math.sigmoid(z_un / tf.math.exp(self.log_temp) + log_alpha)
        z_tilde = tf.math.sigmoid(z_tilde_un / tf.math.exp(self.log_temp) + log_alpha)
        c_phi = self.compute_c_phi(z=z, x=x, params=params)
        c_phi_tilde = self.compute_c_phi(z=z_tilde, x=x, params=params)
        return c_phi, c_phi_tilde, one_hot


class OptIGR(OptVAE):
    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.mu_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.xi_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.dist = IGR_I(mu=self.mu_0, xi=self.xi_0, temp=self.temp)
        self.use_continuous = True
        self.prior_file = hyper['prior_file']

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
        self.dist = IGR_I(mu=mu, xi=xi, temp=self.temp, sample_size=self.sample_size)

    def compute_kl_elements(self, z, params_broad,
                            sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot):
        if self.use_continuous:
            mean, log_var, mu_disc, xi_disc = params_broad
            if sample_from_cont_kl:
                z_norm, _ = z
                kl_norm = sample_kl_norm(z_norm=z_norm, mean=mean, log_var=log_var)
            else:
                kl_norm = calculate_simple_closed_gauss_kl(mean=mean, log_var=log_var)
        else:
            mu_disc, xi_disc = params_broad
            kl_norm = 0.
        if not sample_from_disc_kl:
            if self.model_type == 'IGR_I_Dis':
                log_p_discrete = compute_igr_log_probs(mu_disc, tf.math.exp(xi_disc))
                p_discrete = tf.math.exp(log_p_discrete)
                categories_n = tf.constant(self.n_required + 1, dtype=self.dtype)
                kl_dis = tf.math.reduce_sum(log_p_discrete * p_discrete, axis=(1, 3))
                kl_dis += self.num_of_vars * tf.math.log(categories_n)
            else:
                batch_n, categories_n, sample_size, var_num = z[-1].shape
                one_hot = tf.transpose(tf.one_hot(tf.argmax(z[-1], axis=1),
                                                  depth=categories_n),
                                       perm=[0, 3, 1, 2])
                p_discrete = tf.reduce_mean(one_hot, axis=2, keepdims=True)
                kl_dis = calculate_categorical_closed_kl(
                    log_alpha=p_discrete, normalize=False)
        else:
            if self.stick_the_landing:
                mu_disc = tf.stop_gradient(mu_disc)
                xi_disc = tf.stop_gradient(xi_disc)
            kl_dis = self.compute_discrete_kl(mu_disc, xi_disc, sample_from_disc_kl)
        return kl_norm, kl_dis

    def compute_discrete_kl(self, mu_disc, xi_disc, sample_from_disc_kl):
        mu_disc_prior, xi_disc_prior = self.update_prior_values()
        if sample_from_disc_kl:
            kl_dis = self.compute_sampled_discrete_kl(mu_disc, xi_disc,
                                                      mu_disc_prior, xi_disc_prior)
        else:
            kl_dis = calculate_general_closed_form_gauss_kl(mean_q=mu_disc,
                                                            log_var_q=2. * xi_disc,
                                                            mean_p=mu_disc_prior,
                                                            log_var_p=2. * xi_disc_prior,
                                                            axis=(1, 3))
        return kl_dis

    def compute_sampled_discrete_kl(self, mu_disc, xi_disc,
                                    mu_disc_prior, xi_disc_prior):
        log_qz_x = compute_log_normal_pdf(self.dist.kappa,
                                          mean=mu_disc, log_var=2. * xi_disc)
        log_pz = compute_log_normal_pdf(self.dist.kappa,
                                        mean=mu_disc_prior, log_var=2. * xi_disc_prior)
        kl_dis = tf.reduce_sum(log_qz_x - log_pz, axis=2)
        return kl_dis

    def update_prior_values(self):
        current_batch_n = self.dist.lam.shape[0]
        mu_disc_prior = self.mu_0[:current_batch_n, :, :]
        xi_disc_prior = self.xi_0[:current_batch_n, :, :]
        return mu_disc_prior, xi_disc_prior

    def load_prior_values(self):
        with open(file=self.prior_file, mode='rb') as f:
            parameters = pickle.load(f)

        mu_0 = tf.constant(parameters['mu'], dtype=self.dtype)
        xi_0 = tf.constant(parameters['xi'], dtype=self.dtype)
        categories_n = mu_0.shape[1]
        prior_shape = mu_0.shape
        mu_0 = tf.math.reduce_mean(mu_0, keepdims=True)
        mu_0 = tf.broadcast_to(mu_0, shape=prior_shape)
        xi_0 = tf.math.reduce_mean(xi_0, keepdims=True)
        xi_0 = tf.broadcast_to(xi_0, shape=prior_shape)

        self.mu_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=mu_0, batch_size=self.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.nets.disc_var_num)
        self.xi_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=xi_0, batch_size=self.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.nets.disc_var_num)


class OptIGRDis(OptIGR):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.use_continuous = False
        self.load_prior_values()

    def reparameterize(self, params_broad):
        mu, xi = params_broad
        self.select_distribution(mu, xi)
        self.dist.generate_sample()
        z_discrete = [self.dist.psi]
        return z_discrete


class OptPlanarNF(OptIGR):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)

    def select_distribution(self, mu, xi):
        self.dist = IGR_Planar(mu=mu, xi=xi, planar_flow=self.nets.planar_flow,
                               temp=self.temp, sample_size=self.estimate_kl_w_n)

    def compute_sampled_discrete_kl(self, mu_disc, xi_disc, mu_disc_prior,
                                    xi_disc_prior):
        log_qz_x = compute_log_normal_pdf(self.dist.kappa,
                                          mean=mu_disc, log_var=2. * xi_disc)
        log_pz = compute_log_normal_pdf(self.dist.lam,
                                        mean=mu_disc_prior, log_var=2. * xi_disc_prior)
        kl_dis = tf.reduce_sum(log_qz_x - log_pz, axis=2)
        pf_log_jac_det = calculate_planar_flow_log_determinant(self.dist.kappa,
                                                               self.nets.planar_flow)
        kl_dis = kl_dis + pf_log_jac_det
        return kl_dis


class OptPlanarNFDis(OptIGRDis, OptPlanarNF):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)


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
        self.dist = IGR_SB_Finite(mu, xi, self.temp, self.estimate_kl_w_n)

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
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper,
                         use_continuous=use_continuous)
        self.max_categories = hyper['latent_discrete_n']
        self.threshold = hyper['threshold']
        self.truncation_option = hyper['truncation_option']
        self.prior_file = hyper['prior_file']
        self.quantile = 50
        self.use_continuous = use_continuous

    def select_distribution(self, mu, xi):
        self.dist = IGR_SB(mu, xi, sample_size=self.estimate_kl_w_n,
                           temp=self.temp, threshold=self.threshold)
        self.dist.truncation_option = self.truncation_option
        self.dist.quantile = self.quantile

    def complete_discrete_vector(self):
        batch_size, n_required = self.dist.psi.shape[0], self.dist.psi.shape[1]
        missing = self.max_categories - n_required
        zeros = tf.constant(value=0., dtype=tf.float32,
                            shape=(batch_size, missing, self.sample_size,
                                   self.num_of_vars))
        z_discrete = tf.concat([self.dist.psi, zeros], axis=1)
        return z_discrete


def compute_loss(log_px_z, kl, sample_size=1, run_iwae=False):
    elbo = log_px_z - kl
    if run_iwae:
        elbo = tf.math.reduce_logsumexp(elbo, axis=1)
        loss = -tf.math.reduce_mean(elbo)
        loss += tf.math.log(tf.constant(sample_size, dtype=tf.float32))
    else:
        loss = -tf.math.reduce_mean(elbo)
    return loss


def compute_log_bernoulli_pdf(x, x_logit, sample_size):
    x_broad = tf.repeat(tf.expand_dims(x, 4), axis=4, repeats=sample_size)
    cross_ent = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_broad, logits=x_logit)
    log_px_z = tf.reduce_sum(cross_ent, axis=(1, 2, 3))
    return log_px_z


def compute_log_categorical_pmf(d, log_alpha):
    log_normalized = log_alpha - tf.reduce_logsumexp(log_alpha, axis=1, keepdims=True)
    log_categorical_pmf = tf.math.reduce_sum(d * log_normalized, axis=(1, 3))
    return log_categorical_pmf


def compute_log_categorical_pmf_grad(d, log_alpha):
    normalized = tf.math.softmax(log_alpha, axis=1)
    grad = d - normalized
    return grad


def softplus(x):
    m = tf.maximum(tf.zeros_like(x), x)
    return m + tf.math.log(tf.exp(-m) + tf.math.exp(x - m))


def bernoulli_loglikelihood(b, log_alpha):
    output = b * (-softplus(-log_alpha)) + (1. - b) * (-log_alpha - softplus(-log_alpha))
    return output


def bernoulli_loglikelihood_grad(b, log_alpha):
    sna = tf.math.sigmoid(-log_alpha)
    return b * sna - (1 - b) * (1 - sna)


def compute_log_gaussian_pdf(x, x_logit, sample_size):
    mu, xi = x_logit
    mu = tf.math.sigmoid(mu)
    xi = 1.e-6 + tf.math.softplus(xi)
    pi = 3.141592653589793

    x_broad = tf.broadcast_to(tf.expand_dims(x, 4), shape=x.shape + (sample_size,))

    log_pixel = (- 0.5 * ((x_broad - mu) / xi) ** 2. -
                 0.5 * tf.math.log(2 * pi) - tf.math.log(1.e-8 + xi))
    log_px_z = tf.reduce_sum(log_pixel, axis=[1, 2, 3])
    return log_px_z


def safe_log_prob(x, eps=1.e-8):
    return tf.math.log(tf.clip_by_value(x, eps, 1.0))


def sample_z_tilde_ber(log_alpha, u, eps=1.e-8):
    u_prime = tf.math.sigmoid(-log_alpha)
    v_1 = (u - u_prime) / tf.clip_by_value(1 - u_prime, eps, 1.0)
    v_1 = tf.clip_by_value(v_1, 0, 1)
    v_1 = tf.stop_gradient(v_1)
    v_1 = v_1 * (1 - u_prime) + u_prime
    v_0 = u / tf.clip_by_value(u_prime, eps, 1.0)
    v_0 = tf.clip_by_value(v_0, 0, 1)
    v_0 = tf.stop_gradient(v_0)
    v_0 = v_0 * u_prime

    v = tf.where(u > u_prime, v_1, v_0)
    v = v + tf.stop_gradient(u - v)
    z_tilde_un = log_alpha + safe_log_prob(v) - safe_log_prob(1 - v)
    return z_tilde_un


def sample_z_tilde_cat(one_hot, log_alpha):
    offset = 1.e-20
    bool_one_hot = tf.cast(one_hot, dtype=tf.bool)
    theta = tf.math.softmax(log_alpha, axis=1)
    v = tf.random.uniform(shape=one_hot.shape)
    v_b = tf.where(bool_one_hot, v, 0.)
    v_b = tf.math.reduce_max(v_b, axis=1, keepdims=True)
    v_b = tf.broadcast_to(v_b, shape=v.shape)

    aux1 = -tf.math.log(v + offset) / tf.clip_by_value(theta, 1.e-5, 1.)
    aux2 = tf.math.log(v_b + offset)
    aux = aux1 - aux2
    z_other = -tf.math.log(aux + offset)
    z_b = -tf.math.log(-tf.math.log(v_b + offset) + offset)
    z_tilde = tf.where(bool_one_hot, z_b, z_other)
    return z_tilde


def compute_grad_var_over_batch(relax_grad):
    variance = tf.math.square(relax_grad)
    variance = tf.math.reduce_sum(variance, axis=(1, 2, 3))
    variance = tf.math.reduce_mean(variance)
    return variance


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
    kl_norm = 0.5 * tf.reduce_sum(tf.math.exp(log_var) + tf.math.pow(mean, 2) -
                                  log_var - tf.constant(1.),
                                  axis=1)
    return kl_norm


def calculate_general_closed_form_gauss_kl(mean_q, log_var_q, mean_p,
                                           log_var_p, axis=(1,)):
    var_q = tf.math.exp(log_var_q)
    var_p = tf.math.exp(log_var_p)

    trace_term = tf.reduce_sum(var_q / var_p - 1., axis=axis)
    means_term = tf.reduce_sum(tf.math.pow(mean_q - mean_p, 2) / var_p, axis=axis)
    log_det_term = tf.reduce_sum(log_var_p - log_var_q, axis=axis)
    kl_norm = 0.5 * (trace_term + means_term + log_det_term)
    return kl_norm


def calculate_planar_flow_log_determinant(z, planar_flow):
    log_det = tf.constant(0., dtype=tf.float32)
    nested_layers = int(len(planar_flow.weights) / 3)
    zl = z
    for lay in range(nested_layers):
        pf_layer = planar_flow.get_layer(index=lay)
        w, b, _ = pf_layer.weights
        u = pf_layer.get_u_tilde()
        uTw = tf.math.reduce_sum(u * w, axis=1)
        wTz = tf.math.reduce_sum(w * zl, axis=1)
        h_prime = 1. - tf.math.tanh(wTz + b[0, :, :, :]) ** 2
        log_det -= tf.math.log(tf.math.abs(1 + h_prime * uTw))
        zl = pf_layer.call(zl)
    log_det = tf.reduce_sum(log_det, axis=[-1])
    return log_det


def compute_log_normal_pdf(sample, mean, log_var):
    pi = tf.constant(3.141592653589793, dtype=mean.dtype)
    log2pi = -0.5 * tf.math.log(2 * pi)
    log_exp_sum = -0.5 * (sample - mean) ** 2 * tf.math.exp(-log_var)
    log_normal_pdf = tf.reduce_sum(log2pi + -0.5 * log_var + log_exp_sum, axis=1)
    return log_normal_pdf


def calculate_categorical_closed_kl(log_alpha, normalize=True):
    offset = 1.e-20
    categories_n = tf.constant(log_alpha.shape[1], dtype=log_alpha.dtype)
    log_uniform_inv = tf.math.log(categories_n)
    pi = tf.math.softmax(log_alpha, axis=1) if normalize else log_alpha
    kl_discrete = tf.reduce_sum(
        pi * (tf.math.log(pi + offset) + log_uniform_inv), axis=(1, 3))
    return kl_discrete


def sample_kl_exp_gs(log_psi, log_pi, temp):
    uniform_probs = get_broadcasted_uniform_probs(log_psi.shape, log_pi.dtype)
    temp_prior = tf.constant(0.5, dtype=log_pi.dtype)
    log_pz = compute_log_exp_gs_dist(log_psi=log_psi, logits=tf.math.log(uniform_probs),
                                     temp=temp_prior)
    log_qz_x = compute_log_exp_gs_dist(log_psi=log_psi, logits=log_pi, temp=temp)
    kl_discrete = tf.math.reduce_sum(log_qz_x - log_pz, axis=2)
    return kl_discrete


def get_broadcasted_uniform_probs(shape, dtype):
    batch_n, categories_n, sample_size, disc_var_num = shape
    uniform_probs = tf.constant([1 / categories_n for _ in range(categories_n)],
                                dtype=dtype,
                                shape=(1, categories_n, 1, 1))
    uniform_probs = shape_prior_to_sample_size_and_discrete_var_num(uniform_probs,
                                                                    batch_n,
                                                                    categories_n,
                                                                    sample_size,
                                                                    disc_var_num)
    return uniform_probs


def shape_prior_to_sample_size_and_discrete_var_num(prior_param, batch_size,
                                                    categories_n,
                                                    sample_size, discrete_var_num):
    prior_param = tf.reshape(prior_param, shape=(1, categories_n, 1, 1))
    prior_param = tf.broadcast_to(prior_param, shape=(batch_size, categories_n, 1, 1))
    prior_param = tf.broadcast_to(prior_param, shape=(
        batch_size, categories_n, sample_size, 1))
    prior_param = tf.broadcast_to(prior_param,
                                  shape=(batch_size, categories_n,
                                         sample_size, discrete_var_num))
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
    z_discrete = tf.TensorArray(dtype=original_z.dtype, size=sample_size,
                                element_shape=(batch_n, disc_var_num * disc_latent_n))
    for i in tf.range(sample_size):
        value = tf.reshape(original_z[:, :, i, :],
                           shape=(batch_n, disc_var_num * disc_latent_n))
        z_discrete = z_discrete.write(index=i, value=value)
    z_discrete = tf.transpose(z_discrete.stack(), perm=[1, 2, 0])
    return z_discrete


def make_one_hot(z_dis):
    categories_n = z_dis.shape[1]
    idx = tf.argmax(z_dis, axis=1)
    one_hot = tf.transpose(tf.one_hot(idx, depth=categories_n), perm=[0, 3, 1, 2])
    return one_hot

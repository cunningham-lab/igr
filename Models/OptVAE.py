import pickle
from typing import Tuple
import tensorflow as tf
from os import environ as os_env
from Utils.Distributions import SBDist, compute_log_gs_dist, compute_log_sb_dist, IsoGauSoftMax
from Utils.Distributions import GaussianSoftmaxDist, ExpGSDist, compute_log_exp_gs_dist, GaussianSoftPlus
from Utils.initializations import initialize_mu_and_xi_for_logistic
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'


class OptVAE:

    def __init__(self, model, optimizer, hyper):
        self.model = model
        self.optimizer = optimizer
        self.n_required = hyper['n_required']
        self.run_analytical_kl = hyper['run_analytical_kl']
        self.sample_size = hyper['sample_size']
        self.dataset_name = hyper['dataset_name']

        self.run_ccβvae = hyper['run_ccβvae']
        self.γ = hyper['γ']
        self.discrete_c = tf.constant(0.)
        self.continuous_c = tf.constant(0.)

    def perform_fwd_pass(self, x):
        params = self.model.encode(x)
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
                                 element_shape=(batch_n,) + self.model.image_shape)
        for i in tf.range(sample_size):
            x_logit = x_logit.write(index=i, value=self.model.decode(z[:, :, i])[0])
            # x_logit = x_logit.write(index=i, value=self.model.decode(z[:, :, i]))
        x_logit = tf.transpose(x_logit.stack(), perm=[1, 2, 3, 4, 0])
        return x_logit

    def decode_gaussian(self, z):
        z = reshape_and_stack_z(z=z)
        batch_n, sample_size = z.shape[0], z.shape[2]
        mu = tf.TensorArray(dtype=tf.float32, size=sample_size,
                            element_shape=(batch_n,) + self.model.image_shape)
        xi = tf.TensorArray(dtype=tf.float32, size=sample_size,
                            element_shape=(batch_n,) + self.model.image_shape)
        for i in tf.range(sample_size):
            z_mu, z_xi = self.model.decode(z[:, :, i])
            mu = mu.write(index=i, value=z_mu)
            xi = xi.write(index=i, value=z_xi)
            # z_mu = self.model.decode(z[:, :, i])[0]
            # mu = mu.write(index=i, value=z_mu)
        # mu = tf.transpose(mu.stack(), perm=[1, 2, 3, 4, 0])
        mu = tf.transpose(mu.stack(), perm=[1, 2, 3, 4, 0])
        xi = tf.transpose(xi.stack(), perm=[1, 2, 3, 4, 0])
        x_logit = [mu, xi]
        # x_logit = mu
        return x_logit

    @staticmethod
    def compute_kl_elements(z, params_broad, run_analytical_kl):
        mean, log_var = params_broad
        kl_norm = sample_kl_norm(z_norm=z, mean=mean, log_var=log_var)
        kl_dis = tf.constant(0)
        return kl_norm, kl_dis

    def compute_loss(self, x, x_logit, z, params_broad, run_ccβvae, run_analytical_kl):
        if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
            log_px_z = compute_log_gaussian_pdf(x=x, x_logit=x_logit)
        else:
            log_px_z = compute_log_bernoulli_pdf(x=x, x_logit=x_logit)
        kl_norm, kl_dis = self.compute_kl_elements(z=z, params_broad=params_broad,
                                                   run_analytical_kl=run_analytical_kl)
        kl = kl_norm + kl_dis
        loss = compute_loss(log_px_z=log_px_z, kl_norm=kl_norm, kl_dis=kl_dis,
                            run_ccβvae=run_ccβvae, γ=self.γ,
                            discrete_c=self.discrete_c, continuous_c=self.continuous_c)
        output = (loss, tf.reduce_mean(log_px_z), tf.reduce_mean(kl),
                  tf.reduce_mean(kl_norm), tf.reduce_mean(kl_dis))
        return output

    def compute_losses_from_x_wo_gradients(self, x, run_ccβvae, run_analytical_kl):
        z, x_logit, params_broad = self.perform_fwd_pass(x=x)
        output = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                   run_ccβvae=run_ccβvae, run_analytical_kl=run_analytical_kl)
        loss, recon, kl, kl_norm, kl_dis = output
        return loss, recon, kl, kl_norm, kl_dis

    def compute_gradients(self, x) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x)
            output = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                       run_ccβvae=self.run_ccβvae, run_analytical_kl=self.run_analytical_kl)
            loss, recon, kl, kl_n, kl_d = output
        gradients = tape.gradient(target=loss, sources=self.model.trainable_variables)
        return gradients, loss, recon, kl, kl_n, kl_d

    def monitor_parameter_gradients_at_psi(self, x):
        with tf.GradientTape() as tape:
            params = self.model.encode(x)
            z = self.reparameterize(params_broad=params)
            psi = tf.math.exp(z[-1])
        gradients = tape.gradient(target=psi, sources=params)
        gradients_norm = tf.linalg.norm(gradients, axis=2)

        return gradients_norm

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

# ===========================================================================================================


class OptGS(OptVAE):

    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)
        self.temp = tf.constant(value=hyper['temp'], dtype=tf.float32)

    def reparameterize(self, params_broad):
        mean, log_var, logits = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        gs = ExpGSDist(log_pi=logits, sample_size=self.model.sample_size, temp=self.temp)
        gs.do_reparameterization_trick()
        z_discrete = gs.psi
        self.n_required = z_discrete.shape[1]
        z = [z_norm, z_discrete]
        return z

    def compute_kl_elements(self, z, params_broad, run_analytical_kl):
        if run_analytical_kl:
            kl_norm, kl_dis = self.compute_kl_elements_analytically(params_broad=params_broad)
        else:
            kl_norm, kl_dis = self.compute_kl_elements_via_sample(z=z, params_broad=params_broad)
        return kl_norm, kl_dis

    @staticmethod
    def compute_kl_elements_analytically(params_broad):
        mean, log_var, logits = params_broad
        kl_norm = calculate_kl_norm_via_analytical_formula(mean=mean, log_var=log_var)
        kl_dis = calculate_categorical_closed_kl(log_α=logits)
        return kl_norm, kl_dis

    def compute_kl_elements_via_sample(self, z, params_broad):
        mean, log_var, logits = params_broad
        z_norm, z_discrete = z
        kl_norm = sample_kl_norm(z_norm=z_norm, mean=mean, log_var=log_var)
        kl_dis = sample_kl_gs(ψ=z_discrete, π=tf.math.softmax(logits, axis=1), temp=self.temp)
        return kl_norm, kl_dis


class OptGSDis(OptGS):
    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)

    def reparameterize(self, params_broad):
        logits = params_broad[0]
        gs = ExpGSDist(log_pi=logits, sample_size=self.sample_size, temp=self.temp)
        gs.do_reparameterization_trick()
        self.n_required = gs.psi.shape[1]
        z_discrete = [gs.psi]
        return z_discrete

    def compute_kl_elements(self, z, params_broad, run_analytical_kl):
        if run_analytical_kl:
            kl_norm, kl_dis = self.compute_kl_elements_analytically(params_broad=params_broad)
        else:
            kl_norm, kl_dis = self.compute_kl_elements_via_sample(z=z, params_broad=params_broad)
        return kl_norm, kl_dis

    @staticmethod
    def compute_kl_elements_analytically(params_broad):
        logits = params_broad[0]
        kl_norm = 0.
        kl_dis = calculate_categorical_closed_kl(log_α=logits)
        return kl_norm, kl_dis

    def compute_kl_elements_via_sample(self, z, params_broad):
        logits = params_broad[0]
        z_discrete = z[0]
        kl_norm = 0.
        kl_dis = sample_kl_gs(ψ=z_discrete, π=tf.math.softmax(logits, axis=1), temp=self.temp)
        return kl_norm, kl_dis


# ===========================================================================================================


class OptExpGS(OptVAE):

    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)
        self.temp = tf.constant(value=hyper['temp'], dtype=tf.float32)
        self.log_psi = tf.constant(value=0., dtype=tf.float32)

    def reparameterize(self, params_broad):
        mean, log_var, logits = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        gs = ExpGSDist(log_pi=logits, sample_size=self.sample_size, temp=self.temp)
        gs.do_reparameterization_trick()
        z_discrete = gs.psi
        self.log_psi = gs.log_psi
        self.n_required = z_discrete.shape[1]
        z = [z_norm, z_discrete]
        return z

    def compute_kl_elements(self, z, params_broad, run_analytical_kl):
        mean, log_var, logits = params_broad
        z_norm, _ = z
        kl_norm = sample_kl_norm(z_norm=z_norm, mean=mean, log_var=log_var)
        kl_dis = sample_kl_exp_gs(log_ψ=self.log_psi, log_π=logits, temp=self.temp)
        return kl_norm, kl_dis


class OptExpGSDis(OptExpGS):
    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)

    def reparameterize(self, params_broad):
        gs = ExpGSDist(log_pi=params_broad[0], sample_size=self.sample_size, temp=self.temp)
        gs.do_reparameterization_trick()
        self.log_psi = gs.log_psi
        self.n_required = gs.psi.shape[1]
        z_discrete = [gs.log_psi]
        return z_discrete

    def compute_kl_elements(self, z, params_broad, run_analytical_kl):
        if run_analytical_kl:
            kl_norm, kl_dis = self.compute_kl_elements_analytically(params_broad=params_broad)
        else:
            kl_norm, kl_dis = self.compute_kl_elements_via_sample(params_broad=params_broad)
        return kl_norm, kl_dis

    @staticmethod
    def compute_kl_elements_analytically(params_broad):
        kl_norm = 0.
        kl_dis = calculate_categorical_closed_kl(log_α=params_broad[0])
        return kl_norm, kl_dis

    def compute_kl_elements_via_sample(self, params_broad):
        kl_norm = 0.
        kl_dis = sample_kl_exp_gs(log_ψ=self.log_psi, log_π=params_broad[0], temp=self.temp)
        return kl_norm, kl_dis


# ===========================================================================================================


class OptGauSoftMax(OptVAE):
    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)
        self.temp = tf.constant(value=hyper['temp'], dtype=tf.float32)
        self.prior_file = hyper['prior_file']
        self.mu_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.xi_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.ng = GaussianSoftmaxDist(mu=self.mu_0, xi=self.xi_0)
        self.load_prior_values()

    def reparameterize(self, params_broad):
        mean, log_var, mu, xi = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        self.ng = GaussianSoftmaxDist(mu=mu, xi=xi, temp=self.temp, sample_size=self.sample_size)
        self.ng.do_reparameterization_trick()
        z_discrete = self.ng.psi
        self.n_required = z_discrete.shape[1]
        z = [z_norm, z_discrete]
        return z

    def compute_kl_elements(self, z, params_broad, run_analytical_kl):
        if run_analytical_kl:
            kl_norm, kl_dis = self.compute_kl_elements_analytically(params_broad=params_broad)
        else:
            kl_norm, kl_dis = self.compute_kl_elements_via_sample(z=z, params_broad=params_broad)
        return kl_norm, kl_dis

    def compute_kl_elements_analytically(self, params_broad):
        mean, log_var, μ0, ξ0 = params_broad
        kl_norm = calculate_kl_norm_via_analytical_formula(mean=mean, log_var=log_var)
        current_batch_n = self.ng.lam.shape[0]
        ξ1 = self.xi_0[:current_batch_n, :, :]
        μ1 = self.mu_0[:current_batch_n, :, :]
        kl_dis = calculate_kl_norm_via_general_analytical_formula(mean_0=μ0, log_var_0=2 * ξ0,
                                                                  mean_1=μ1, log_var_1=2. * ξ1)
        return kl_norm, kl_dis

    def compute_kl_elements_via_sample(self, z, params_broad):
        mean, log_var, μ, ξ = params_broad
        z_norm, z_discrete = z
        kl_norm = sample_kl_norm(z_norm=z_norm, mean=mean, log_var=log_var)
        kl_dis = self.sample_kl_sb()
        return kl_norm, kl_dis

    def sample_kl_sb(self):
        current_batch_n = self.ng.lam.shape[0]
        log_pz = compute_log_normal_pdf(sample=self.temp * self.ng.lam,
                                        mean=self.mu_0[:current_batch_n, :self.ng.n_required, :],
                                        log_var=self.xi_0[:current_batch_n, :self.ng.n_required, :])
        log_qz_x = compute_log_normal_pdf(sample=self.temp * self.ng.lam, mean=self.ng.mu, log_var=self.ng.xi)
        kl_sb = log_qz_x - log_pz
        return kl_sb

    def load_prior_values(self):
        shape = (self.model.batch_size, self.model.disc_latent_n, self.sample_size, self.model.disc_var_num)
        self.mu_0, self.xi_0 = initialize_mu_and_xi_for_logistic(shape=shape)


class OptGauSoftPlus(OptGauSoftMax):
    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)
        self.ng = GaussianSoftPlus(mu=self.mu_0, xi=self.xi_0, temp=self.temp)

    def reparameterize(self, params_broad):
        mean, log_var, mu, xi = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        self.ng = GaussianSoftPlus(mu=mu, xi=xi, temp=self.temp)
        self.ng.do_reparameterization_trick()
        z_discrete = self.ng.psi
        self.n_required = z_discrete.shape[1]
        z = [z_norm, z_discrete]
        return z


class OptGauSoftMaxDis(OptGauSoftMax):

    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)

    def reparameterize(self, params_broad):
        mu, xi = params_broad
        self.ng = GaussianSoftmaxDist(mu=mu, xi=xi, temp=self.temp, sample_size=self.sample_size)
        self.ng.do_reparameterization_trick()
        z_discrete = [self.ng.log_psi]
        return z_discrete

    def compute_kl_elements(self, z, params_broad, run_analytical_kl):
        if run_analytical_kl:
            kl_norm, kl_dis = self.compute_kl_elements_analytically(params_broad=params_broad)
        else:
            kl_norm, kl_dis = self.compute_kl_elements_via_sample(z=z, params_broad=params_broad)
        return kl_norm, kl_dis

    def compute_kl_elements_analytically(self, params_broad):
        μ0, ξ0 = params_broad
        kl_norm = 0.
        current_batch_n = self.ng.lam.shape[0]
        ξ1 = self.xi_0[:current_batch_n, :, :]
        μ1 = self.mu_0[:current_batch_n, :, :]
        kl_dis = calculate_kl_norm_via_general_analytical_formula(mean_0=μ0, log_var_0=2 * ξ0,
                                                                  mean_1=μ1, log_var_1=2. * ξ1,
                                                                  axis=(1, 3))
        return kl_norm, kl_dis

    def compute_kl_elements_via_sample(self, z, params_broad):
        kl_norm = 0.
        μ, ξ = params_broad
        current_batch_n = self.ng.lam.shape[0]
        log_pz = compute_log_normal_pdf(sample=self.temp * self.ng.lam,
                                        mean=self.mu_0[:current_batch_n, :, :, :],
                                        log_var=2. * self.xi_0[:current_batch_n, :, :, :])
        log_qz_x = compute_log_normal_pdf(sample=self.temp * self.ng.lam,
                                          mean=μ, log_var=2. * ξ)
        kl_dis = log_qz_x - log_pz
        kl_dis = tf.reduce_sum(kl_dis, axis=2)
        return kl_norm, kl_dis


class OptPlanarNFDis(OptGauSoftMaxDis):

    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)

    def reparameterize(self, params_broad):
        mu, xi = params_broad
        epsilon = tf.random.normal(shape=mu.shape)
        self.ng = GaussianSoftmaxDist(mu=mu, xi=xi, temp=self.temp, sample_size=self.sample_size)
        sigma = tf.math.exp(xi)
        self.ng.lam = self.model.planar_flow(mu + sigma * epsilon)
        self.ng.log_psi = self.ng.lam - tf.math.reduce_logsumexp(self.ng.lam, axis=1, keepdims=True)
        # psi = tf.math.softmax(lam / self.temp, axis=1)
        z_discrete = [self.ng.log_psi]
        return z_discrete


class OptIsoGauSoftMaxDis(OptGauSoftMaxDis):

    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)

    def reparameterize(self, params_broad):
        mu = params_broad[0]
        self.ng = IsoGauSoftMax(mu=mu, temp=self.temp, sample_size=self.sample_size)
        self.ng.do_reparameterization_trick()
        z_discrete = [self.ng.psi]
        return z_discrete


class OptGauSoftPlusDis(OptGauSoftMaxDis):

    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)

    def reparameterize(self, params_broad):
        mu, xi = params_broad
        self.ng = GaussianSoftPlus(mu=mu, xi=xi, temp=self.temp, sample_size=self.sample_size,
                                   noise_type='trunc_normal')
        self.ng.do_reparameterization_trick()
        z_discrete = [self.ng.log_psi]
        # z_discrete = [self.ng.psi]
        return z_discrete


class OptSBVAE(OptVAE):

    def __init__(self, model, optimizer, hyper):
        super().__init__(model=model, optimizer=optimizer, hyper=hyper)
        self.temp = tf.constant(value=hyper['temp'], dtype=tf.float32)
        self.max_categories = hyper['latent_discrete_n']
        self.threshold = hyper['threshold']
        self.prior_file = hyper['prior_file']
        self.temp_min = hyper['temp']
        self.truncation_option = hyper['truncation_option']
        self.quantile = 70
        self.mu_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1))
        self.xi_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1))
        self.sb = SBDist(mu=self.mu_0, xi=self.xi_0)
        self.load_prior_values()

    def reparameterize(self, params_broad):
        mean, log_var, μ, ξ = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        self.sb = SBDist(mu=μ, xi=ξ, sample_size=self.sample_size, temp=self.temp, threshold=self.threshold)
        self.sb.truncation_option = self.truncation_option
        self.sb.quantile = self.quantile
        self.sb.do_reparameterization_trick()
        self.n_required = self.sb.psi.shape[1]
        z_discrete = self.complete_discrete_vector(psi=self.sb.psi)

        z = [z_norm, z_discrete]
        return z

    def compute_kl_elements(self, z, params_broad, run_analytical_kl):
        if run_analytical_kl:
            kl_norm, kl_dis = self.compute_kl_elements_analytically(params_broad=params_broad)
        else:
            kl_norm, kl_dis = self.compute_kl_elements_via_sample(z=z, params_broad=params_broad)
        return kl_norm, kl_dis

    def compute_kl_elements_analytically(self, params_broad):
        mean, log_var, μ0, ξ0 = params_broad
        kl_norm = calculate_kl_norm_via_analytical_formula(mean=mean, log_var=log_var)
        current_batch_n = self.sb.lam.shape[0]
        ξ1 = self.xi_0[:current_batch_n, :, :]
        μ1 = self.mu_0[:current_batch_n, :, :]
        kl_dis = calculate_kl_norm_via_general_analytical_formula(mean_0=μ0, log_var_0=2 * ξ0,
                                                                  mean_1=μ1, log_var_1=2. * ξ1)
        return kl_norm, kl_dis

    def compute_kl_elements_via_sample(self, z, params_broad):
        mean, log_var, μ, ξ = params_broad
        z_norm, z_discrete = z
        kl_norm = sample_kl_norm(z_norm=z_norm, mean=mean, log_var=log_var)
        kl_dis = self.sample_kl_sb()
        return kl_norm, kl_dis

    def sample_kl_sb(self):
        # TODO: take out of class once the function below is implemented
        current_batch_n = self.sb.lam.shape[0]
        sigma_0 = tf.math.exp(self.xi_0[:current_batch_n, :self.sb.n_required, :, 0])
        epsilon_0 = (self.sb.delta - self.mu_0[:current_batch_n, :self.sb.n_required, :, 0]) / sigma_0
        log_pz = compute_log_sb_dist(lam=self.sb.lam, kappa=self.sb.kappa, log_jac=self.sb.log_jac,
                                     temp=tf.constant(self.temp_min),
                                     sigma=sigma_0, epsilon=epsilon_0)
        log_qz_x = self.sb.compute_log_sb_dist()
        kl_sb = log_qz_x - log_pz
        return kl_sb

    # -------------------------------------------------------------------------------------------------------
    # Utils
    def complete_discrete_vector(self, psi):
        batch_size, n_required = psi.shape[0], psi.shape[1]
        missing = self.max_categories - n_required
        zeros = tf.constant(value=0., dtype=tf.float32, shape=(batch_size, missing, self.sample_size, 1))
        z_discrete = tf.concat([psi, zeros], axis=1)
        return z_discrete

    def load_prior_values(self):
        with open(file=self.prior_file, mode='rb') as f:
            parameters = pickle.load(f)

        mu_0 = tf.constant(parameters['mu'], dtype=tf.float32)
        xi_0 = tf.constant(parameters['xi'], dtype=tf.float32)
        categories_n = mu_0.shape[1]

        self.mu_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=mu_0, batch_size=self.model.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.model.disc_var_num)
        self.xi_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=xi_0, batch_size=self.model.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.model.disc_var_num)


# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Additional functions
# ===========================================================================================================
def compute_loss(log_px_z, kl_norm, kl_dis, run_ccβvae=False,
                 γ=tf.constant(1.), discrete_c=tf.constant(0.), continuous_c=tf.constant(0.)):
    if run_ccβvae:
        loss = -tf.reduce_mean(log_px_z - γ * tf.math.abs(kl_norm - continuous_c)
                               - γ * tf.math.abs(kl_dis - discrete_c))
    else:
        kl = kl_norm + kl_dis
        elbo = tf.reduce_mean(log_px_z - kl)
        loss = -elbo
    return loss


def compute_log_bernoulli_pdf(x, x_logit):
    batch_size, image_size, sample_size = x_logit.shape[0], x_logit.numpy().shape[1:4], x_logit.shape[4]
    x_w_extra_col = tf.reshape(x, shape=(batch_size,) + image_size + (1,))
    x_broad = tf.broadcast_to(x_w_extra_col, shape=(batch_size,) + image_size + (sample_size,))
    cross_ent = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_broad, logits=x_logit)
    log_px_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return log_px_z


def compute_log_gaussian_pdf(x, x_logit):
    mu, xi = x_logit
    # mu = x_logit
    mu = tf.math.sigmoid(mu)
    xi = 1.e-6 + tf.math.softplus(xi)
    pi = 3.141592653589793

    batch_size, image_size, sample_size = mu.shape[0], mu.numpy().shape[1:4], mu.shape[4]
    x_w_extra_col = tf.reshape(x, shape=(batch_size,) + image_size + (1,))
    x_broad = tf.broadcast_to(x_w_extra_col, shape=(batch_size,) + image_size + (sample_size,))

    # log_pixel = - 0.5 * (x_broad - mu) ** 2. - 0.5 * tf.math.log(2 * pi)
    # log_px_z = tf.reduce_sum(log_pixel, axis=[1, 2, 3])

    log_pixel = - 0.5 * ((x_broad - mu) / xi) ** 2. - 0.5 * tf.math.log(2 * pi) - tf.math.log(1.e-8 + xi)
    log_px_z = tf.reduce_sum(log_pixel, axis=[1, 2, 3])
    return log_px_z


def sample_normal(mean, log_var):
    ε = tf.random.normal(shape=mean.shape)
    z_norm = mean + tf.math.exp(log_var * 0.5) * ε
    return z_norm


def sample_kl_norm(z_norm, mean, log_var):
    log_pz = compute_log_normal_pdf(sample=z_norm, mean=0., log_var=0.)
    log_qz_x = compute_log_normal_pdf(sample=z_norm, mean=mean, log_var=log_var)
    kl_norm = log_qz_x - log_pz
    return kl_norm


def calculate_kl_norm_via_analytical_formula(mean, log_var):
    kl_norm = 0.5 * tf.reduce_sum(tf.math.exp(log_var) + tf.math.pow(mean, 2) - log_var - tf.constant(1.),
                                  axis=1)
    return kl_norm


def calculate_kl_norm_via_general_analytical_formula(mean_0, log_var_0, mean_1, log_var_1, axis=(1,)):
    var_0 = tf.math.exp(log_var_0)
    var_1 = tf.math.exp(log_var_1)

    trace_term = tf.reduce_sum(var_0 / var_1 - 1., axis=axis)
    means_term = tf.reduce_sum(tf.math.pow(mean_0 - mean_1, 2) / var_1, axis=axis)
    log_det_term = tf.reduce_sum(log_var_1 - log_var_0, axis=axis)
    kl_norm = 0.5 * (trace_term + means_term + log_det_term)
    return kl_norm


def compute_log_normal_pdf(sample, mean, log_var):
    π = 3.141592653589793
    log2pi = -0.5 * tf.math.log(2 * π)
    log_exp_sum = -0.5 * (sample - mean) ** 2 * tf.math.exp(-log_var)
    log_normal_pdf = tf.reduce_sum(log2pi + -0.5 * log_var + log_exp_sum, axis=1)
    return log_normal_pdf


def sample_kl_gs(ψ, π, temp):
    uniform_probs = get_broadcasted_uniform_probs(shape=ψ.shape)
    log_pz = compute_log_gs_dist(psi=ψ, logits=uniform_probs, temp=temp)
    log_qz_x = compute_log_gs_dist(psi=ψ, logits=π, temp=temp)
    kl_discrete = tf.math.reduce_sum(log_qz_x - log_pz, axis=2)
    return kl_discrete


def get_broadcasted_uniform_probs(shape):
    batch_n, categories_n, sample_size, disc_var_num = shape
    uniform_probs = tf.constant([1 / categories_n for _ in range(categories_n)], dtype=tf.float32,
                                shape=(1, categories_n, 1, 1))
    uniform_probs = tf.broadcast_to(uniform_probs, shape=(batch_n, categories_n, 1, 1))
    uniform_probs = tf.broadcast_to(uniform_probs, shape=(batch_n, categories_n, sample_size, 1))
    uniform_probs = tf.broadcast_to(uniform_probs, shape=(batch_n, categories_n, sample_size, disc_var_num))
    return uniform_probs


def calculate_categorical_closed_kl(log_α):
    ς = 1.e-20
    categories_n = tf.constant(log_α.shape[1], dtype=tf.float32)
    log_uniform_inv = tf.math.log(categories_n)
    π = tf.math.softmax(log_α, axis=1)
    kl_discrete = tf.reduce_sum(π * tf.math.log(π + ς), axis=1) + log_uniform_inv
    return kl_discrete


def sample_kl_exp_gs(log_ψ, log_π, temp):
    uniform_probs = get_broadcasted_uniform_probs(shape=log_ψ.shape)
    log_pz = compute_log_exp_gs_dist(log_psi=log_ψ, logits=tf.math.log(uniform_probs), temp=temp)
    log_qz_x = compute_log_exp_gs_dist(log_psi=log_ψ, logits=log_π, temp=temp)
    kl_discrete = tf.math.reduce_sum(log_qz_x - log_pz, axis=2)
    return kl_discrete


def shape_prior_to_sample_size_and_discrete_var_num(prior_param, batch_size, categories_n,
                                                    sample_size, discrete_var_num):
    prior_param = tf.reshape(prior_param, shape=(1, categories_n, 1, 1))
    prior_param = tf.broadcast_to(prior_param, shape=(batch_size, categories_n, 1, 1))
    prior_param = tf.broadcast_to(prior_param, shape=(batch_size, categories_n, sample_size, 1))
    prior_param = tf.broadcast_to(prior_param,
                                  shape=(batch_size, categories_n, sample_size, discrete_var_num))
    return prior_param


def flatten_discrete_variables(original_z):
    batch_n, disc_latent_n, sample_size, disc_var_num = original_z.shape
    z_discrete = tf.reshape(original_z, shape=(batch_n, disc_var_num * disc_latent_n, sample_size))
    return z_discrete


def reshape_and_stack_z(z):
    if len(z) > 1:
        z = tf.concat(z, axis=1)
        z = flatten_discrete_variables(original_z=z)
    else:
        z = flatten_discrete_variables(original_z=z[0])
    return z

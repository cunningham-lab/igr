import unittest
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from Utils.Distributions import iterative_sb
from Tests.TestDistributions import calculate_eta_from_kappa
from Models.OptVAE import OptDLGMM


class TestDLGMM(unittest.TestCase):

    def test_kumaraswamy_reparam(self):
        log_a = tf.random.normal(shape=(2, 4, 1, 1))
        log_b = tf.random.normal(shape=(2, 4, 1, 1))
        with tf.GradientTape() as tape:
            tape.watch([log_a, log_b])
            kumar = tfpd.Kumaraswamy(concentration0=tf.math.exp(log_a),
                                     concentration1=tf.math.exp(log_b))
            z_kumar = kumar.sample()
        grad = tape.gradient(target=z_kumar, sources=[log_a, log_b])
        print('\nTEST: Kumaraswamy Reparameterization Gradient')
        self.assertTrue(expr=grad is not None)

    def test_sb(self):
        test_tolerance = 1.e-6
        z_kumar = tf.constant([[0.1, 0.2, 0.3, 0.4],
                               [0.2, 0.3, 0.4, 0.1]])
        z_kumar = tf.expand_dims(tf.expand_dims(z_kumar, axis=-1), axis=-1)
        approx = iterative_sb(z_kumar)
        ans = tf.constant(calculate_eta_from_kappa(z_kumar.numpy()), dtype=z_kumar.dtype)
        remainder = 1. - tf.reduce_sum(ans, axis=1, keepdims=True)
        ans = tf.concat((ans, remainder), axis=1)
        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Stick-Break Kumar')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def setUp(self):
        self.hyper = {'latent_norm_n': 0, 'num_of_norm_param': 0, 'num_of_norm_var': 0,
                      'save_every': 50, 'sample_size_testing': 1 * int(1.e0),
                      'dtype': tf.float32, 'sample_size': 1, 'check_every': 50,
                      'epochs': 300, 'learning_rate': 1 * 1.e-4, 'batch_n': 100,
                      'num_of_discrete_var': 20, 'sample_from_disc_kl': True,
                      'stick_the_landing': True, 'test_with_one_hot': False,
                      'dataset_name': 'mnist',
                      'model_type': 'linear', 'temp': 1.0,
                      'sample_from_cont_kl': True}

    def test_loss(self):
        test_tolerance = 1.e-2
        tf.random.set_seed(seed=21)
        batch_n, n_required, sample_size, dim = 2, 4, 10, 3
        shape = (batch_n, n_required, sample_size, dim)
        log_a = tf.random.normal(shape=(batch_n, n_required - 1, 1, 1))
        log_b = tf.random.normal(shape=log_a.shape)
        kumar = tfpd.Kumaraswamy(concentration0=tf.math.exp(log_a),
                                 concentration1=tf.math.exp(log_b))
        z_kumar = kumar.sample()
        z_norm = tf.random.normal(shape=shape)
        mean = tf.random.normal(shape=shape)
        log_var = tf.zeros_like(z_norm)
        pi = iterative_sb(z_kumar)
        x = tf.random.uniform(shape=(batch_n, 4, 4, 1))
        x_logit = tf.random.normal(shape=(batch_n, 4, 4, 1, sample_size, n_required))
        self.hyper['n_required'] = n_required
        self.hyper['sample_size'] = sample_size
        optvae = OptDLGMM(nets=[], optimizer=[], hyper=self.hyper)
        z = [z_kumar, z_norm]
        params_broad = [log_a, log_b, mean, log_var]
        optvae.batch_size, optvae.n_required = batch_n, n_required
        optvae.sample_size, optvae.num_of_vars = sample_size, dim
        optvae.mu_prior = optvae.create_separated_prior_means()
        optvae.log_var_prior = tf.zeros_like(z_norm)

        approx = optvae.compute_loss(x, x_logit, z, params_broad,
                                     True, True, True, True)
        # ans = -calculate_kumar_entropy(log_a, log_b)
        ans = compute_kld(log_a, log_b)
        ans += calculate_log_qz_x(z_norm, pi, mean, log_var)
        ans -= calculate_log_px_z(x, x_logit, pi)
        ans -= calculate_log_pz(z_norm, pi, optvae.mu_prior, optvae.log_var_prior)
        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Loss')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def test_log_px_z(self):
        test_tolerance = 1.e-6
        batch_n, sample_size = 2, 10
        pi = tf.constant([[0.1, 0.2, 0.3, 0.4]])
        pi = tf.expand_dims(tf.expand_dims(pi, axis=-1), axis=-1)
        n_required = pi.shape[1]
        x = tf.random.uniform(shape=(batch_n, 4, 4, 1))
        x_logit = tf.random.normal(shape=(batch_n, 4, 4, 1, sample_size, n_required))
        self.hyper['n_required'] = n_required
        self.hyper['sample_size'] = sample_size

        optvae = OptDLGMM(nets=[], optimizer=[], hyper=self.hyper)
        approx = optvae.compute_log_px_z(x, x_logit, pi)
        ans = calculate_log_px_z(x, x_logit, pi)

        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Reconstruction')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def test_log_pz(self):
        test_tolerance = 1.e-6
        batch_n, n_required, sample_size, dim = 2, 4, 10, 3
        pi = tf.constant([[0.1, 0.2, 0.3, 0.4]])
        pi = tf.expand_dims(tf.expand_dims(pi, axis=-1), axis=-1)
        n_required = pi.shape[1]
        z = tf.random.normal(shape=(batch_n, n_required, sample_size, dim))
        self.hyper['n_required'] = n_required
        self.hyper['sample_size'] = sample_size
        optvae = OptDLGMM(nets=[], optimizer=[], hyper=self.hyper)
        optvae.batch_size, optvae.n_required = batch_n, n_required
        optvae.sample_size, optvae.num_of_vars = sample_size, dim
        optvae.mu_prior = optvae.create_separated_prior_means()
        optvae.log_var_prior = tf.zeros_like(z)
        approx = optvae.compute_log_pz(z, pi)
        ans = calculate_log_pz(z, pi, optvae.mu_prior, optvae.log_var_prior)

        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Normal Prior')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def test_kld(self):
        test_tolerance = 1.e-5
        batch_n, n_required = 2, 4
        tf.random.set_seed(seed=21)
        # log_a = tf.constant([[0., -1., 1., 2.0, -2.0]])
        # log_a = tf.expand_dims(tf.expand_dims(log_a, axis=-1), axis=-1)
        # log_b = tf.constant([[0., -1., 1., 2.0, -2.0]])
        # log_b = tf.expand_dims(tf.expand_dims(log_b, axis=-1), axis=-1)
        log_a = tf.random.normal(shape=(batch_n, n_required, 1, 1))
        log_b = tf.random.normal(shape=log_a.shape)
        self.hyper['n_required'] = log_a.shape[1]
        optvae = OptDLGMM(nets=[], optimizer=[], hyper=self.hyper)

        approx = optvae.compute_kld(log_a, log_b)
        # ans = -calculate_kumar_entropy(log_a, log_b)
        ans = compute_kld(log_a, log_b)
        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Kumaraswamy KL')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def test_log_qz_x(self):
        test_tolerance = 1.e-6
        batch_n, n_required, sample_size, dim = 2, 4, 10, 3
        pi = tf.constant([[0.1, 0.2, 0.3, 0.4]])
        pi = tf.expand_dims(tf.expand_dims(pi, axis=-1), axis=-1)
        n_required = pi.shape[1]
        z = tf.random.normal(shape=(batch_n, n_required, sample_size, dim))
        self.hyper['n_required'] = n_required
        self.hyper['sample_size'] = sample_size
        optvae = OptDLGMM(nets=[], optimizer=[], hyper=self.hyper)
        mean = tf.random.normal(shape=(batch_n, n_required, sample_size, dim))
        log_var = tf.zeros_like(z)
        approx = optvae.compute_log_qz_x(z, pi, mean, log_var)
        ans = calculate_log_qz_x(z, pi, mean, log_var)

        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Normal A Posterior')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)


def get_case01():
    # TODO: finish the case
    output = 0
    return output


def calculate_log_px_z(x, x_logit, pi):
    sample_size = x_logit.shape[4]
    n_required = x_logit.shape[5]
    # x_broad = tf.repeat(tf.expand_dims(x, 4), axis=4, repeats=pi.shape[1])
    # x_broad = tf.reshape(x_broad, shape=(16, 4))
    # x_logit = tf.reshape(x_logit, shape=(16, 4))
    x_broad = tf.repeat(tf.expand_dims(x, -1), axis=-1, repeats=sample_size)
    x_broad = tf.repeat(tf.expand_dims(x_broad, -1), axis=-1, repeats=n_required)
    dist = tfpd.Bernoulli(logits=x_logit)
    log_px_z = dist.log_prob(x_broad)
    log_px_z = tf.reduce_sum(log_px_z, axis=(1, 2, 3))
    log_px_z = tf.reduce_mean(log_px_z, axis=1)
    log_px_z = tf.reduce_sum(pi[:, :, 0, 0] * log_px_z, axis=1)
    log_px_z = tf.reduce_mean(log_px_z, axis=0)
    return log_px_z


def calculate_log_pz(z, pi, mu_prior, log_var_prior):
    loc = mu_prior
    scale = tf.math.exp(0.5 * log_var_prior)
    dist = tfpd.Normal(loc, scale)
    log_pz = pi * tf.reduce_sum(dist.log_prob(z), axis=3, keepdims=True)
    log_pz = tf.reduce_mean(log_pz, axis=2, keepdims=True)
    log_pz = tf.reduce_mean(tf.reduce_sum(log_pz, axis=(1, 2, 3)))
    return log_pz


def calculate_kumar_entropy(log_a, log_b):
    a, b = tf.math.exp(log_a), tf.math.exp(log_b)
    # a, b = tf.math.softplus(log_a), tf.math.softplus(log_b)
    n_required = log_a.shape[1]
    ans = 0.
    for k in range(n_required):
        dist = tfpd.Kumaraswamy(concentration0=a[:, k, 0, 0],
                                concentration1=b[:, k, 0, 0])
        ans += dist.entropy()
    ans = tf.reduce_mean(ans)
    return ans


def calculate_log_qz_x(z, pi, mean, log_var):
    dist = tfpd.Normal(loc=mean, scale=tf.math.exp(0.5 * log_var))
    aux = tf.reduce_prod(tf.math.exp(dist.log_prob(z)), axis=3, keepdims=True)
    qz_x = tf.reduce_sum(pi * aux, axis=1, keepdims=True)
    log_qz_x = tf.reduce_mean(tf.math.log(qz_x), axis=2, keepdims=True)
    log_qz_x = tf.reduce_mean(tf.math.reduce_sum(log_qz_x, axis=(1, 2, 3)))

    return log_qz_x


def beta_fn(a, b):
    output = (tf.math.exp(tf.math.lgamma(a) + tf.math.lgamma(b) -
                          tf.math.lgamma(a + b)))
    return output


def compute_kld(log_a, log_b):
    alpha = tf.ones_like(log_a)
    beta = tf.ones_like(log_b)
    # a, b = tf.math.softplus(log_a), tf.math.softplus(log_b)
    a, b = tf.math.exp(log_a), tf.math.exp(log_b)
    ab = tf.math.multiply(a, b)
    a_inv = tf.pow(a, -1)
    b_inv = tf.pow(b, -1)

    # compute taylor expansion for E[log (1-v)] term
    kl = tf.math.multiply(tf.pow(1 + ab, -1), beta_fn(a_inv, b))
    for idx in range(10):
        kl += tf.math.multiply(tf.pow(idx + 2 + ab, -1),
                               beta_fn(tf.math.multiply(idx + 2., a_inv), b))
    kl = tf.math.multiply(tf.math.multiply(beta - 1, b), kl)

    kl += tf.math.multiply(tf.math.truediv(a - alpha, a), -0.57721 -
                           tf.math.digamma(b) - b_inv)
    # add normalization constants
    kl += tf.math.log(ab) + tf.math.log(beta_fn(alpha, beta))

    kl += tf.math.truediv(-(b - 1), b)
    kl = tf.reduce_mean(kl, axis=2, keepdims=True)
    kl = tf.reduce_sum(kl, axis=(1, 2, 3))
    kl = tf.reduce_mean(kl, axis=0)
    return kl


if __name__ == '__main__':
    unittest.main()

import tensorflow as tf
from Models.VAENet import RelaxCovNet


class RELAX:

    def __init__(self, loss_f, lr, shape):
        self.loss_f = loss_f
        self.loss = tf.constant(0.)
        self.optimizer_param = tf.optimizers.Adam(lr)
        self.optimizer_var = tf.optimizers.Adam(lr)

        self.shape = shape
        self.batch_n = shape[0]
        self.categories_n = shape[1]
        self.sample_size = shape[2]
        self.num_of_vars = shape[3]
        cov_net_shape = (self.categories_n, self.sample_size, self.num_of_vars)
        self.relax_cov = RelaxCovNet(cov_net_shape)

        self.log_alpha = tf.Variable(tf.constant(0., shape=shape), name='log_alpha', trainable=True)
        self.log_temp = tf.Variable(0., name='log_temp', trainable=True)
        self.eta = tf.Variable(1., name='eta', trainable=True)
        self.con_net_vars = self.relax_cov.net.trainable_variables + [self.log_temp] + [self.eta]
        self.one_hot = tf.constant(0., shape=shape)

    def get_relax_ingredients(self):
        u = tf.random.uniform(shape=self.shape)
        z_un = self.log_alpha + safe_log_prob(u) - safe_log_prob(1 - u)
        self.one_hot = tf.cast(tf.stop_gradient(z_un) > 0, dtype=tf.float32)

        z_tilde_un = sample_z_tilde_ber(self.log_alpha, self.one_hot)
        c_phi = self.compute_c_phi(z_un)
        c_phi_tilde = self.compute_c_phi(z_tilde_un)
        return c_phi, c_phi_tilde

    def compute_c_phi(self, z_un):
        r = tf.math.reduce_mean(self.relax_cov.net(z_un))
        z = tf.math.sigmoid(z_un / tf.math.exp(self.log_temp))
        c_phi = self.loss_f(z) + r
        return c_phi

    def compute_log_pmf_grad(self):
        grad = bernoulli_loglikelihood_grad(self.one_hot, self.log_alpha)
        return grad

    def compute_rebar_gradients_and_loss(self):
        c_phi, c_phi_tilde = self.get_relax_ingredients()
        self.loss = self.loss_f(self.one_hot)
        log_qz_x_grad_theta = self.compute_log_pmf_grad()
        c_phi_diff_grad_theta = tf.gradients(c_phi - c_phi_tilde, self.log_alpha)[0]
        self.relax = self.compute_relax_grad(c_phi_tilde, log_qz_x_grad_theta,
                                             c_phi_diff_grad_theta)
        variance = compute_relax_grad_variance(self.relax)
        self.cov_net_grad = tf.gradients(variance, self.con_net_vars)
        return (self.relax, self.cov_net_grad), self.loss

    def compute_relax_grad(self, c_phi_tilde, log_qz_x_grad, c_phi_diff_grad_theta):
        diff = self.loss - self.eta * c_phi_tilde
        relax_grad = diff * log_qz_x_grad
        relax_grad += self.eta * c_phi_diff_grad_theta
        return relax_grad

    def apply_gradients(self, grads):
        relax, cov_net_grad = grads
        self.optimizer_param.apply_gradients(zip([relax], [self.log_alpha]))
        self.optimizer_var.apply_gradients(zip(cov_net_grad, self.con_net_vars))


def compute_relax_grad_variance(relax_grad):
    variance = tf.math.square(relax_grad)
    variance = tf.math.reduce_sum(variance, axis=(1, 2, 3))
    variance = tf.math.reduce_mean(variance)
    return variance


def toy_loss(one_hot, t=0.4):
    loss = tf.reduce_sum((one_hot - t) ** 2, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss)
    return loss


def toy_loss_2(one_hot):
    t = tf.constant([1., 0.4, 0.0])
    t = tf.expand_dims(t, 0)
    t = tf.expand_dims(t, 1)
    t = tf.expand_dims(t, 2)
    loss = tf.reduce_sum((one_hot - t) ** 2, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss)
    return loss


def softplus(x):
    m = tf.maximum(tf.zeros_like(x), x)
    return m + tf.math.log(tf.exp(-m) + tf.math.exp(x - m))


def bernoulli_loglikelihood(b, log_alpha):
    output = b * (-softplus(-log_alpha)) + (1. - b) * (-log_alpha - softplus(-log_alpha))
    return output


def bernoulli_loglikelihood_grad(b, log_alpha):
    sna = tf.math.sigmoid(-log_alpha)
    return b * sna - (1 - b) * (1 - sna)


def safe_log_prob(x, eps=1.e-8):
    return tf.math.log(tf.clip_by_value(x, eps, 1.0))


def sample_z_tilde_ber(log_alpha, one_hot, eps=1.e-8):
    # TODO: add testing for this function
    v = tf.random.uniform(shape=log_alpha.shape)
    theta = tf.math.sigmoid(log_alpha)
    v_0 = v * (1 - theta)
    v_1 = v * theta + (1 - theta)
    v_tilde = tf.where(one_hot > 0, v_1, v_0)

    z_tilde_un = log_alpha + safe_log_prob(v_tilde) - safe_log_prob(1 - v_tilde)
    return z_tilde_un


# def sample_z_tilde_ber(log_alpha, eps=1.e-8):
#     u = tf.random.uniform(shape=log_alpha.shape)
#     u_prime = tf.math.sigmoid(-log_alpha)
#     v_1 = (u - u_prime) / tf.clip_by_value(1 - u_prime, eps, 1.0)
#     v_1 = tf.clip_by_value(v_1, 0, 1)
#     v_1 = v_1 * (1 - u_prime) + u_prime
#     v_0 = u / tf.clip_by_value(u_prime, eps, 1.0)
#     v_0 = tf.clip_by_value(v_0, 0, 1)
#     v_0 = v_0 * u_prime
#
#     v = tf.where(u > u_prime, v_1, v_0)
#     z_tilde_un = log_alpha + safe_log_prob(v) - safe_log_prob(1 - v)
#     return z_tilde_un

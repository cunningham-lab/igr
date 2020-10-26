import unittest
import numpy as np
import tensorflow as tf
from Models.OptVAE import calculate_simple_closed_gauss_kl, calculate_categorical_closed_kl
from Models.OptVAE import calculate_general_closed_form_gauss_kl
from Models.OptVAE import calculate_planar_flow_log_determinant
from Models.OptVAE import sample_z_tilde_cat
from Models.OptVAE import sample_z_tilde_ber
from Models.OptVAE import compute_log_categorical_pmf
from Models.OptVAE import compute_log_categorical_pmf_grad
from Models.OptVAE import bernoulli_loglikelihood
from Models.OptVAE import bernoulli_loglikelihood_grad
from Models.VAENet import create_nested_planar_flow
from Tests.TestVAENet import calculate_pf_log_det_np_all
from Utils.Distributions import project_to_vertices_via_softmax_pp
from Utils.Distributions import project_to_vertices
from Utils.Distributions import compute_igr_log_probs


class TestOptandDist(unittest.TestCase):

    def test_gradient_concat(self):
        test_tolerance = 1.e-7
        batch_n, categories_n, sample_size, num_of_var = 1, 9, 1, 1
        shape = (batch_n, categories_n, sample_size, num_of_var)
        x = tf.random.normal(shape=(batch_n, 3))
        w1 = tf.random.normal(shape=(3, categories_n))
        w2 = tf.random.normal(shape=(3, categories_n))
        b1 = tf.random.normal(shape=(batch_n, categories_n))
        b2 = tf.random.normal(shape=(batch_n, categories_n))
        encoder_vars = [w1, b1, w2, b2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(encoder_vars)
            mu = tf.reshape(x @ w1 + b1, shape)
            xi = tf.reshape(x @ w2 + b2, shape)
            tape.watch([mu, xi])
            sigma = tf.math.exp(tf.clip_by_value(xi, -50., 50.))
            z_un = mu + sigma * tf.random.normal(shape=mu.shape)
            z = project_to_vertices_via_softmax_pp(z_un)
            one_hot = project_to_vertices(z, categories_n=categories_n + 1)
            log_probs = compute_igr_log_probs(mu, sigma)
            log_pmf = tf.reduce_sum(tf.reduce_sum(one_hot * log_probs, axis=1), axis=(1, 1))

        log_pmf_g = tape.gradient(target=log_pmf, sources=encoder_vars)
        log_cat_g = compute_log_categorical_grads(one_hot, encoder_vars, x, shape)

        print(f'\nTEST: Gradient Concatenation')
        for idx, grad in enumerate(log_pmf_g):
            diff = np.linalg.norm(log_cat_g[idx] - grad) / np.linalg.norm(grad)
            print(f'\nDiff {diff:1.3e}')
            self.assertTrue(expr=diff < test_tolerance)

    def test_bernoulli_grad(self):
        tolerance = 1.e-7
        batch_n, categories_n, sample_size, num_of_vars = 4, 6, 1, 5

        log_alpha_np = np.random.normal(size=(batch_n, categories_n, sample_size, num_of_vars))
        log_alpha = tf.constant(log_alpha_np, dtype=tf.float32)
        one_hot = tf.transpose(tf.one_hot(tf.argmax(log_alpha, axis=1), depth=categories_n),
                               perm=[0, 3, 1, 2])

        grad = bernoulli_loglikelihood_grad(one_hot, log_alpha)

        with tf.GradientTape() as tape:
            tape.watch(log_alpha)
            log_cat_pmf = bernoulli_loglikelihood(one_hot, log_alpha)

        grad_auto = tape.gradient(target=log_cat_pmf, sources=log_alpha)
        print(f'\nTEST: Bernoulli Gradient')
        diff = np.linalg.norm(grad - grad_auto) / np.linalg.norm(grad)
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(diff < tolerance)

    def test_log_categorical_grad(self):
        tolerance = 1.e-7
        batch_n, categories_n, sample_size, num_of_vars = 4, 6, 1, 5

        log_alpha_np = np.random.normal(size=(batch_n, categories_n, sample_size, num_of_vars))
        log_alpha = tf.constant(log_alpha_np, dtype=tf.float32)
        one_hot = tf.transpose(tf.one_hot(tf.argmax(log_alpha, axis=1), depth=categories_n),
                               perm=[0, 3, 1, 2])

        grad = compute_log_categorical_pmf_grad(one_hot, log_alpha)

        with tf.GradientTape() as tape:
            tape.watch(log_alpha)
            log_cat_pmf = compute_log_categorical_pmf(one_hot, log_alpha)

        grad_auto = tape.gradient(target=log_cat_pmf, sources=log_alpha)
        print(f'\nTEST: Categorical Gradient')
        diff = np.linalg.norm(grad - grad_auto) / np.linalg.norm(grad)
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(diff < tolerance)

    def test_sample_z_tilde_ber(self):
        tolerance = 1.e-2
        sample_size, num_of_vars = int(1.e4), 1
        temp = tf.constant(1.)

        one_hot_np = np.array([[1.], [0.]])
        one_hot_np = broadcast_to_shape(one_hot_np, sample_size, num_of_vars)
        one_hot = tf.constant(one_hot_np, dtype=tf.float32)

        log_alpha_np = np.array([[2.], [-4.]])
        log_alpha_np = broadcast_to_shape(log_alpha_np, sample_size, num_of_vars)
        log_alpha = tf.constant(log_alpha_np, dtype=tf.float32)

        z_tilde_un = sample_z_tilde_ber(one_hot=one_hot, log_alpha=log_alpha)
        z_tilde = tf.math.sigmoid(z_tilde_un / temp)
        z_tilde_dist = tf.reduce_mean(z_tilde, axis=2).numpy()

        z = sample_conditional_all(log_alpha_np, one_hot_np, sample_conditional_ber)
        z_dist_np = np.mean(z, axis=2)

        print(f'\nTEST: z_tilde Sampling Correct Conditonal Bernoulli')
        diff = np.linalg.norm(z_tilde_dist - z_dist_np) / np.linalg.norm(z_dist_np)
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(diff < tolerance)

    def test_sample_z_tilde_cat(self):
        tolerance = 1.e-2
        sample_size, num_of_vars = int(1.e3), 2
        temp = tf.constant(1.)

        one_hot_np = np.array([[1., 0., 0.], [0., 1., 0.]])
        one_hot_np = broadcast_to_shape(one_hot_np, sample_size, num_of_vars)
        one_hot = tf.constant(one_hot_np, dtype=tf.float32)

        # log_alpha_np = np.array([[0., -1., -2.], [0.5, -5, -4.]]) # non-intuitive case
        log_alpha_np = np.array([[2., -1., 0.], [-4., 0.5, -5.]])  # easy case
        log_alpha_np = broadcast_to_shape(log_alpha_np, sample_size, num_of_vars)
        log_alpha = tf.constant(log_alpha_np, dtype=tf.float32)

        z_tilde = sample_z_tilde_cat(one_hot, log_alpha)
        z_tilde = tf.math.softmax(z_tilde / temp, axis=1)
        correct_max = tf.math.argmax(z_tilde, axis=1)
        correct_max = tf.math.reduce_mean(correct_max, axis=1).numpy()
        z_tilde_dist = tf.reduce_mean(z_tilde, axis=2).numpy()

        z = sample_conditional_all(log_alpha_np, one_hot_np, sample_conditional_gs)
        z_max = np.argmax(z, axis=1)
        correct_max_np = np.mean(z_max, axis=1)
        z_dist_np = np.mean(z, axis=2)

        print(f'\nTEST: z_tilde Sampling Correct One-Hot')
        diff = np.linalg.norm(correct_max - correct_max_np) / np.linalg.norm(correct_max_np)
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(diff < tolerance)

        print(f'\nTEST: z_tilde Sampling Correct Distribution')
        diff = np.linalg.norm(z_tilde_dist - z_dist_np) / np.linalg.norm(z_dist_np)
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(diff < tolerance)

    def test_planar_flow_log_det_broadcasted(self):
        tolerance = 1.e-5
        batch_n = 1
        sample_size = 1
        categories_n = 4
        input_shape = batch_n, categories_n, sample_size, batch_n
        nested_layers, var_num = 2, 1
        z = np.array([1 / np.sqrt(categories_n) for _ in np.arange(categories_n)])
        z = prepare_example(z, input_shape)
        planar_flow = create_nested_planar_flow(nested_layers, categories_n, var_num)
        approx = calculate_planar_flow_log_determinant(z, planar_flow).numpy()
        w_np, u_np, b_np = get_pf_layers_weights(planar_flow)
        ans = calculate_pf_log_det_np_all(z, w_np, u_np, b_np)
        diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
        print(f'\nTEST: TF Planar Flow Log Determinant')
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_planar_flow_log_det_autodiff(self):
        tolerance = 1.e-5
        batch_n, sample_size, categories_n, var_num = 1, 1, 4, 1
        nested_layers = 1
        input_shape = batch_n, categories_n, sample_size, batch_n
        z = np.array([1 / np.sqrt(categories_n) for _ in np.arange(categories_n)])
        z = prepare_example(z, input_shape)

        planar_flow = create_nested_planar_flow(nested_layers, categories_n, var_num)
        approx = calculate_planar_flow_log_determinant(z, planar_flow).numpy()
        with tf.GradientTape() as tape:
            tape.watch(z)
            output = planar_flow.call(z)
        jac = tape.jacobian(target=output, sources=z)
        jac = jac[0, :, 0, 0, 0, :, 0, 0]
        ans = -np.log(tf.linalg.det(jac))
        diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
        print(f'\nTEST: TF Planar Flow Autodiff Log Determinant')
        print(f'\nDiff {diff:1.3e}')
        self.assertTrue(expr=diff < tolerance)

    def test_calculate_kl_norm_via_analytical_formula(self):
        test_tolerance = 1.e-5
        cases = []
        samples_n, num_of_vars = 1, 2
        mu = broadcast_to_shape(np.array([[1., -1., 0.15, 0.45, -0.3],
                                          [-0.1, 0.98, 0.02, -1.4, 0.35],
                                          [0., 0., 2., 2., 0.]]),
                                samples_n=samples_n, num_of_vars=num_of_vars)
        log_sigma2 = broadcast_to_shape(np.array(
            [[-1.7261764, 0.1970883, -0.05951275, 0.43101027, 1.00751897],
             [-1.55425, -0.0337, 1.22609, -0.19088, 0.9577],
             [0., 0., 0., 0., 0.]]), samples_n=samples_n, num_of_vars=num_of_vars)
        cases.append((mu, log_sigma2))

        samples_n, batch_n, categories_n = 2, 68, 15
        mu = broadcast_to_shape(np.random.normal(size=(batch_n, categories_n)),
                                samples_n=samples_n, num_of_vars=num_of_vars)
        log_sigma2 = broadcast_to_shape(np.random.lognormal(size=(batch_n, categories_n)),
                                        samples_n=samples_n, num_of_vars=num_of_vars)
        cases.append((mu, log_sigma2))

        print(f'\nTEST: Closed-form Gaussian KL (sigma vector)')
        for mu, log_sigma2 in cases:
            kl_norm_ans = calculate_kl_norm(mu=mu, sigma2=np.exp(log_sigma2))
            kl_norm = calculate_simple_closed_gauss_kl(mean=tf.constant(mu, dtype=tf.float32),
                                                       log_var=tf.constant(log_sigma2,
                                                                           dtype=tf.float32))
            relative_diff = np.linalg.norm((kl_norm.numpy() - kl_norm_ans) / kl_norm_ans)
            print(f'\nDiff {relative_diff:1.3e}')
            self.assertTrue(expr=relative_diff < test_tolerance)

    def test_calculate_kl_gs_via_discrete_formula(self):
        test_tolerance = 1.e-5
        samples_n = 1
        num_of_vars = 1
        log_alpha = broadcast_to_shape(np.array([[1., -1., 0.15, 0.45, -0.3],
                                                 [-0.1, 0.98, 0.02, -1.4, 0.35]]),
                                       samples_n=samples_n, num_of_vars=num_of_vars)
        kl_discrete_ans = calculate_kl_discrete(alpha=tf.math.softmax(log_alpha, axis=1))
        kl_discrete_ans = np.sum(kl_discrete_ans, axis=-1)
        kl_discrete = calculate_categorical_closed_kl(
            log_alpha=tf.constant(log_alpha, dtype=tf.float32))
        relative_diff = np.linalg.norm((kl_discrete.numpy() - kl_discrete_ans) / kl_discrete_ans)
        print(f'\nTEST: KL GS Discrete')
        print(f'\nDiff {relative_diff:1.3e}')
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_calculate_kl_norm_via_general_analytical_formula(self):
        test_tolerance = 1.e-10
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
        kl_norm = calculate_general_closed_form_gauss_kl(mean_q=mean_0, log_var_q=log_var_0,
                                                         mean_p=mean_1, log_var_p=log_var_1)
        relative_diff = np.linalg.norm((kl_norm.numpy() - kl_norm_ans))
        print(f'\nTEST: Closed-form Gaussian KL (General)')
        print(f'\nDiff {relative_diff:1.3e}')
        self.assertTrue(expr=relative_diff < test_tolerance)

        self.assertTrue(expr=np.isclose(kl_norm.numpy()[-1, 0, 0], 0))


def calculate_kl_norm(mu, sigma2):
    categories_n = mu.shape[1]
    kl_norm = np.zeros(shape=mu.shape)
    for i in range(categories_n):
        kl_norm[:, i, :] = 0.5 * (sigma2[:, i, :] + mu[:, i, :] ** 2 - np.log(sigma2[:, i, :]) - 1)
    return np.sum(kl_norm, axis=1)


def calculate_kl_norm_general_from_vectors(mean_0, log_var_0, mean_1, log_var_1):
    batch_n, categories_n, samples_n, num_of_var = mean_0.shape
    kl_norm = np.zeros(shape=(batch_n, samples_n, num_of_var))
    for i in range(batch_n):
        for j in range(samples_n):
            for r in range(num_of_var):
                cov_0 = np.diag(np.exp(log_var_0[i, :, j, r]))
                cov_1 = np.diag(np.exp(log_var_1[i, :, j, r]))
                kl_norm[i, j, r] = calculate_kl_norm_general(mean_0=mean_0[i, :, j, r],
                                                             cov_0=cov_0,
                                                             mean_1=mean_1[i, :, j, r],
                                                             cov_1=cov_1)
    return kl_norm


def calculate_kl_norm_general(mean_0, cov_0, mean_1, cov_1):
    categories_n = mean_0.shape[0]
    trace_term = np.trace(np.linalg.inv(cov_1) @ cov_0)
    means_term = (mean_1 - mean_0).T @ np.linalg.inv(cov_1) @ (mean_1 - mean_0)
    log_det_term = np.log(np.linalg.det(cov_1) / np.linalg.det(cov_0))
    kl_norm = 0.5 * (trace_term + means_term + log_det_term - categories_n)
    return kl_norm


def calculate_kl_discrete(alpha):
    categories_n = alpha.shape[1]
    kl_discrete = np.zeros(shape=alpha.shape)
    for i in range(categories_n):
        kl_discrete[:, i, :] = alpha[:, i, :] * (np.log(alpha[:, i, :]) - np.log(1 / categories_n))
    return np.sum(kl_discrete, axis=1)


def prepare_example(z, input_shape, make_tf=True):
    batch_n, categories_n, sample_size, var_num = input_shape
    z = np.reshape(z, newshape=(1, categories_n, 1, var_num))
    z = np.broadcast_to(z, shape=input_shape)
    z = tf.constant(z, dtype=tf.float32) if make_tf else z
    return z


def get_pf_layers_weights(planar_flow):
    nested_layers = int(len(planar_flow.weights) / 3)
    pf_layer = planar_flow.get_layer(index=0)
    w0, b0, u0 = pf_layer.weights
    w_np = np.zeros(shape=(w0.shape + (nested_layers,)))
    b_np = np.zeros(shape=(b0.shape + (nested_layers,)))
    u_np = np.zeros(shape=(u0.shape + (nested_layers,)))
    w_np[:, :, :, :, 0] = w0.numpy()
    b_np[:, :, :, :, 0] = b0.numpy()
    u_np[:, :, :, :, 0] = u0.numpy()
    for l in range(1, nested_layers):
        pf_layer = planar_flow.get_layer(index=l)
        w0, b0, u0 = pf_layer.weights
        w_np[:, :, :, :, l] = w0.numpy()
        b_np[:, :, :, :, l] = b0.numpy()
        u_np[:, :, :, :, l] = u0.numpy()
    return w_np, u_np, b_np


def broadcast_to_shape(v, samples_n, num_of_vars):
    original_shape = v.shape
    v = np.reshape(v, newshape=original_shape + (1, 1))
    v = np.broadcast_to(v, shape=original_shape + (samples_n, 1))
    v = np.broadcast_to(v, shape=original_shape + (samples_n, num_of_vars))
    return v


def sample_conditional_all(log_alpha, one_hot, sample_conditional):
    batch_n, categories_n, sample_size, num_of_vars = log_alpha.shape
    z = np.zeros(log_alpha.shape)
    for b in range(batch_n):
        for n in range(num_of_vars):
            z[b, :, :, n] = sample_conditional(log_alpha[b, :, 0, n],
                                               sample_size=sample_size,
                                               one_hot=one_hot[b, :, 0, n])
    return z


def sample_conditional_ber(log_alpha, sample_size, one_hot):
    k, total, z = 0, 0, []
    while total < sample_size:
        unif = np.random.uniform(size=1)
        logistic = np.log(unif) - np.log(1 - unif)
        sample = log_alpha + logistic
        sample_exp = np.exp(sample)
        sigmoid = sample_exp / (1 + sample_exp)
        if np.round(sigmoid) == one_hot:
            z.append(sigmoid)
            k += 1
            total += 1
    output = np.array(z)
    return output.T


def sample_conditional_gs(log_alpha, sample_size, one_hot):
    categories_n = log_alpha.shape[0]
    argmax = np.argmax(one_hot)
    k = 0
    total = 0
    z = []
    while total < sample_size:
        unif = np.random.uniform(size=categories_n)
        gumbel = -np.log(-np.log(unif))
        sample = log_alpha + gumbel
        if np.argmax(sample) == argmax:
            sample_exp = np.exp(sample)
            norm = np.sum(sample_exp)
            sigmoid = sample_exp / norm
            z.append(sigmoid)
            k += 1
            total += 1
    output = np.array(z)
    return output.T


@tf.function()
def compute_log_categorical_grads(z, encoder_vars, x, shape):
    w1, b1, w2, b2 = encoder_vars
    mu = tf.reshape(x @ w1 + b1, shape)
    xi = tf.reshape(x @ w2 + b2, shape)
    params = [mu, xi]
    sigma = tf.math.exp(tf.clip_by_value(xi, -50., 50.))
    log_probs = compute_igr_log_probs(mu, sigma)
    log_qz_x_grad = tf.gradients(log_probs, params, grad_ys=z)
    grad = tf.gradients(params, encoder_vars, grad_ys=log_qz_x_grad)
    return grad


if __name__ == '__main__':
    unittest.main()

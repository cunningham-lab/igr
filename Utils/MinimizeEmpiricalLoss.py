import time
import numpy as np
import tensorflow as tf
from Utils.Distributions import compute_gradients, apply_gradients, generate_sample
from Utils.initializations import initialize_mu_and_xi_for_logistic
from Utils.mmd import compute_mmd2u
from Utils.general import setup_logger
logger = setup_logger(log_file_name='./Log/discrete.log')


class MinimizeEmpiricalLoss:

    def __init__(self, learning_rate: float, temp_init: float, temp_final: float, moments_diff: float = 9.2,
                 pool=None, sample_size: int = int(1.e3), max_iterations: int = int(1.e4),
                 run_kl: bool = True, tolerance: float = 1.e-5,
                 model_type: str = 'logit', type_temp_schedule: str = 'anneal'):

        self.learning_rate = learning_rate
        self.moments_diff = moments_diff
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.sample_size = sample_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.model_type = model_type
        self.type_temp_schedule = type_temp_schedule
        self.run_kl = run_kl

        self.iteration = 0
        self.training_time = 0
        self.iter_time = 0
        self.total_samples_for_moment_evaluation = 1
        self.mean_progress = 10
        self.mean_loss = 10
        self.mean_n_required = 0
        self.diff = 10
        self.mmd = 10
        self.check_every = 10
        self.anneal_rate = 0.00003
        self.threshold = 0.9
        self.temp_schedule = np.linspace(temp_final, temp_init, num=max_iterations)
        self.temp = tf.constant(value=temp_init, dtype=tf.float32)
        self.pool = pool
        self.q_samples = np.zeros(shape=self.total_samples_for_moment_evaluation)
        self.categories_n = 0
        self.params = []
        self.loss_iter = np.zeros(shape=max_iterations)
        self.n_required_iter = np.zeros(shape=max_iterations)
        self.mmd_bandwidth = np.sqrt(9 / 2)
        self.run_iteratively = False

    def set_variables(self, params):
        self.params = params
        self.categories_n = params[0].shape[1]

    def optimize_model(self, mean_p, var_p, probs, p_samples):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        continue_training = True
        self.total_samples_for_moment_evaluation = 100
        self.q_samples = np.zeros(shape=self.total_samples_for_moment_evaluation)
        t0 = time.time()
        while continue_training:
            current_iter_time = time.time()
            loss, n_required = self.take_gradient_step_and_compute_loss(optimizer=optimizer, probs=probs)
            self.iter_time += time.time() - current_iter_time

            self.loss_iter[self.iteration] = loss.numpy()
            self.n_required_iter[self.iteration] = n_required
            self.update_temperature()

            self.evaluate_progress(loss_iter=self.loss_iter, n_required_iter=self.n_required_iter)
            self.iteration += 1
            continue_training = self.determine_continuation()
        self.training_time = time.time() - t0
        logger.info(f'\nTraining took: {self.training_time:6.1f} sec')
        self.check_moments_convergence(mean_p=mean_p, var_p=var_p)
        choose_100_at_random = np.random.choice(a=self.total_samples_for_moment_evaluation,
                                                size=100, replace=False)
        self.mmd = compute_mmd2u(p_samples=p_samples[choose_100_at_random], bandwidth=self.mmd_bandwidth,
                                 q_samples=self.q_samples[choose_100_at_random])
        logger.info(f'Diff {self.diff: 2.2f}')
        logger.info(f'MMD_u^2 {self.mmd: 2.2e}')

    def take_gradient_step_and_compute_loss(self, optimizer, probs):
        grad, loss, n_required = compute_gradients(params=self.params, temp=self.temp,
                                                   probs=probs, dist_type=self.model_type,
                                                   sample_size=self.sample_size, threshold=self.threshold,
                                                   run_iteratively=self.run_iteratively,
                                                   run_kl=self.run_kl)
        apply_gradients(optimizer=optimizer, gradients=grad, variables=self.params)
        return loss, n_required

    def update_temperature(self):
        if self.type_temp_schedule == 'anneal':
            current_temp = np.maximum(self.temp_init * np.exp(-self.anneal_rate * self.iteration),
                                      self.temp_final)
        elif self.type_temp_schedule == 'linear':
            current_temp = self.temp_schedule[-self.iteration]
        elif self.type_temp_schedule == 'constant':
            current_temp = self.temp
        else:
            current_temp = 0
            print('Temp Error - No valid type selected')
        self.temp = tf.constant(current_temp, dtype=tf.float32)

    def evaluate_progress(self, loss_iter, n_required_iter):
        if (self.iteration % self.check_every == 0) & (self.iteration > self.check_every):
            self.check_mean_loss_to_previous_iterations(loss_iter=loss_iter, n_required_iter=n_required_iter)
            # self.check_mean_progress_to_previous_iterations(loss_iter=loss_iter)
            # self.check_moments_convergence(mean_p=mean_p, var_p=var_p)

            logger.info(f'Iter {self.iteration:4d} || '
                        f'Loss {self.mean_loss:2.3e} || '
                        f'N Required {int(self.mean_n_required):4d} || '
                        f'R {self.mean_progress: 2.1e} || '
                        f'Diff {self.diff: 2.2f}')

    def check_mean_loss_to_previous_iterations(self, loss_iter, n_required_iter):
        new_window = self.iteration - self.check_every
        self.mean_loss = np.mean(loss_iter[new_window: self.iteration])
        self.mean_n_required = np.mean(n_required_iter[new_window: self.iteration])

    def check_mean_progress_to_previous_iterations(self, loss_iter):
        past_window = self.iteration - 2 * self.check_every
        new_window = self.iteration - self.check_every
        sufficient_iterations_have_passed = past_window >= 0
        if sufficient_iterations_have_passed:
            recent_mean_improvement = np.mean(loss_iter[new_window: self.iteration])
            last_mean_improvement = np.mean(loss_iter[past_window:new_window])

            self.mean_progress = (np.abs(recent_mean_improvement - last_mean_improvement))

    def check_moments_convergence(self, mean_p, var_p):
        for i in range(self.q_samples.shape[0]):
            self.q_samples[i] = generate_sample(sample_size=1, params=self.params, dist_type=self.model_type,
                                                temp=self.temp, threshold=self.threshold)[0, :]

        self.diff = np.sqrt(((np.mean(self.q_samples) - mean_p) ** 2 +
                             (np.var(self.q_samples) - var_p) ** 2))

    def determine_continuation(self):
        continue_training = ((self.iteration < self.max_iterations) &
                             (self.mean_progress > self.tolerance) &
                             (self.diff > self.moments_diff) &
                             (np.abs(self.mean_loss) > self.tolerance))
        return continue_training


def get_initial_params_for_model_type(model_type, shape):
    batch_size, categories_n, sample_size, num_of_vars = shape
    if model_type == 'ExpGS':
        uniform_probs = np.array([1 / categories_n for _ in range(categories_n)])
        pi = tf.constant(value=np.log(uniform_probs), dtype=tf.float32,
                         shape=(batch_size, categories_n, 1, 1))
        pi_init = tf.constant(pi.numpy().copy(), dtype=tf.float32,
                              shape=(batch_size, categories_n, 1, 1))
        params = [tf.Variable(initial_value=pi)]
        params_init = [tf.Variable(initial_value=pi_init)]
    elif model_type == 'sb':
        shape2 = (batch_size, categories_n, sample_size)
        mu, xi = initialize_mu_and_xi_for_logistic(shape2, seed=21)
        mu_init, xi_init = tf.constant(mu.numpy().copy()), tf.constant(xi.numpy().copy())
        params = [mu, xi]
        params_init = [mu_init, xi_init]

    elif model_type == 'GauSoftMax' or model_type == 'GauSoftPlus':
        mu, xi = initialize_mu_and_xi_for_logistic(shape, seed=21)
        mu_init, xi_init = tf.constant(mu.numpy().copy()), tf.constant(xi.numpy().copy())
        params = [mu, xi]
        params_init = [mu_init, xi_init]

        # mu = np.random.normal(size=shape)
        # xi = np.random.normal(size=shape)
        # mu = tf.constant(value=mu, dtype=tf.float32)
        # xi = tf.constant(value=xi, dtype=tf.float32)
        # mu_init, xi_init = tf.constant(mu.numpy().copy()), tf.constant(xi.numpy().copy())
        # params = [tf.Variable(initial_value=mu), tf.Variable(initial_value=xi)]
        # params_init = [tf.Variable(initial_value=mu_init), tf.Variable(initial_value=xi_init)]
    elif model_type == 'IsoGauSoftMax' or model_type == 'IsoGauSoftPlus':
        mu = np.random.normal(size=shape)
        mu_init = tf.constant(value=mu.copy(), dtype=tf.float32)
        mu = tf.constant(value=mu, dtype=tf.float32)
        params = [tf.Variable(initial_value=mu)]
        params_init = [tf.Variable(initial_value=mu_init)]
    else:
        raise RuntimeError
    return params, params_init


def obtain_results_from_minimizer(minimizer):
    iteration = minimizer.iteration
    time_taken = minimizer.iter_time
    diff = minimizer.diff
    mmd = minimizer.mmd
    n_required = minimizer.mean_n_required
    return iteration, time_taken, diff, mmd, n_required


def update_results(results_to_update, minimizer, dist_case, run_case, cat_case: int = 1):
    tracks = obtain_results_from_minimizer(minimizer=minimizer)
    for idx, var in enumerate(results_to_update):
        if minimizer.model_type == 'ExpGS':
            var[cat_case, dist_case, run_case] = tracks[idx]
        else:
            var[dist_case, run_case] = tracks[idx]

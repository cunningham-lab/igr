import time
import numpy as np
import tensorflow as tf
from Utils.Distributions import compute_gradients, apply_gradients
from Utils.general import initialize_mu_and_xi_for_logistic, initialize_mu_and_xi_equally, setup_logger

logger = setup_logger(log_file_name='./Log/discrete.log')


class MinimizeEmpiricalLoss:

    def __init__(self, params, learning_rate, temp, sample_size=int(1.e3), max_iterations=int(1.e4),
                 run_kl=True, tolerance=1.e-5, model_type='IGR_I', threshold=0.9):

        self.params = params
        self.learning_rate = learning_rate
        self.temp = tf.constant(value=temp, dtype=tf.float32)
        self.sample_size = sample_size
        self.max_iterations = max_iterations
        self.run_kl = run_kl
        self.tolerance = tolerance
        self.model_type = model_type
        self.threshold = threshold

        self.iteration = 0
        self.training_time = 0
        self.iter_time = 0
        self.mean_loss = 10
        self.mean_n_required = 0
        self.check_every = 10
        self.loss_iter = np.zeros(shape=max_iterations)
        self.n_required_iter = np.zeros(shape=max_iterations)
        self.run_iteratively = False

    def optimize_model(self, probs):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        continue_training = True
        t0 = time.time()
        while continue_training:
            current_iter_time = time.time()
            loss, n_required = self.take_gradient_step_and_compute_loss(optimizer=optimizer, probs=probs)
            self.iter_time += time.time() - current_iter_time

            self.loss_iter[self.iteration] = loss.numpy()
            self.n_required_iter[self.iteration] = n_required

            self.evaluate_progress(loss_iter=self.loss_iter, n_required_iter=self.n_required_iter)
            self.iteration += 1
            continue_training = self.determine_continuation()
        self.training_time = time.time() - t0
        logger.info(f'\nTraining took: {self.training_time:6.1f} sec')

    def take_gradient_step_and_compute_loss(self, optimizer, probs):
        grad, loss, n_required = compute_gradients(params=self.params, temp=self.temp,
                                                   probs=probs, dist_type=self.model_type,
                                                   sample_size=self.sample_size, threshold=self.threshold,
                                                   run_iteratively=self.run_iteratively,
                                                   run_kl=self.run_kl)
        apply_gradients(optimizer=optimizer, gradients=grad, variables=self.params)
        return loss, n_required

    def evaluate_progress(self, loss_iter, n_required_iter):
        if (self.iteration % self.check_every == 0) & (self.iteration > self.check_every):
            self.check_mean_loss_to_previous_iterations(loss_iter=loss_iter, n_required_iter=n_required_iter)

            logger.info(f'Iter {self.iteration:4d} || '
                        f'Loss {self.mean_loss:2.3e} || '
                        f'N Required {int(self.mean_n_required):4d}')

    def check_mean_loss_to_previous_iterations(self, loss_iter, n_required_iter):
        new_window = self.iteration - self.check_every
        self.mean_loss = np.mean(loss_iter[new_window: self.iteration])
        self.mean_n_required = np.mean(n_required_iter[new_window: self.iteration])

    def determine_continuation(self):
        continue_training = ((self.iteration < self.max_iterations) &
                             (np.abs(self.mean_loss) > self.tolerance))
        return continue_training


def get_initial_params_for_model_type(model_type, shape):
    batch_size, categories_n, sample_size, num_of_vars = shape
    if model_type == 'GS':
        uniform_probs = np.array([1 / categories_n for _ in range(categories_n)])
        pi = tf.constant(value=np.log(uniform_probs), dtype=tf.float32,
                         shape=(batch_size, categories_n, 1, 1))
        pi_init = tf.constant(pi.numpy().copy(), dtype=tf.float32,
                              shape=(batch_size, categories_n, 1, 1))
        # noinspection PyArgumentList
        params = [tf.Variable(initial_value=pi)]
        # noinspection PyArgumentList
        params_init = [tf.Variable(initial_value=pi_init)]
    elif model_type in ['IGR_I', 'IGR_SB', 'IGR_SB_Finite']:
        shape_igr = (batch_size, categories_n - 1, sample_size, num_of_vars)
        if model_type == 'IGR_I':
            mu, xi = initialize_mu_and_xi_equally(shape_igr)
        else:
            mu, xi = initialize_mu_and_xi_for_logistic(shape_igr, seed=21)
        params_init = [tf.constant(mu.numpy().copy()), tf.constant(xi.numpy().copy())]
        params = [mu, xi]
    else:
        raise RuntimeError
    return params, params_init


def obtain_results_from_minimizer(minimizer):
    iteration = minimizer.iteration
    time_taken = minimizer.iter_time
    n_required = minimizer.mean_n_required
    return iteration, time_taken, n_required


def update_results(results_to_update, minimizer, dist_case, run_case, cat_case: int = 1):
    tracks = obtain_results_from_minimizer(minimizer=minimizer)
    for idx, var in enumerate(results_to_update):
        if minimizer.model_type == 'GS':
            var[cat_case, dist_case, run_case] = tracks[idx]
        else:
            var[dist_case, run_case] = tracks[idx]

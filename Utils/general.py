import datetime
import logging
import time
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
import numba
import seaborn as sns
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import tensorflow as tf
import pickle
from typing import Tuple
from scipy.stats import poisson, binom, geom, nbinom
from Utils.Distributions import generate_sample
from os import environ as os_env
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Approximating Discrete Distributions
#  =====================================================================================================
def offload_case(case):
    run_against = case['run_against']
    categories_n = case['categories_n']
    categories_n_list = case['categories_n_list']
    y_lim_max = case['y_lim_max']
    x_lim_max = case['x_lim_max']
    return run_against, categories_n, categories_n_list, y_lim_max, x_lim_max


def sample_from_approx(params, params_init, temp, model_type, p_samples, samples_plot_n, threshold,
                       planar_flow=None):
    temp = tf.constant(temp, dtype=tf.float32)
    q_samples = np.zeros(shape=samples_plot_n)
    q_samples_init = np.zeros(shape=samples_plot_n)
    for sample_id in range(samples_plot_n):
        q_samples[sample_id] = generate_sample(sample_size=1, params=params, temp=temp,
                                               threshold=threshold,
                                               dist_type=model_type, planar_flow=planar_flow)
        q_samples_init[sample_id] = generate_sample(sample_size=1, params=params_init,
                                                    dist_type=model_type,
                                                    temp=temp, threshold=threshold,
                                                    planar_flow=planar_flow)
    print(f'{model_type}')
    print(f'Mean {np.mean(q_samples):4.2f} || '
          f'Var {np.var(q_samples):4.2f} || '
          f'Std {np.std(q_samples):4.2f}'
          f'\nMin: {np.min(q_samples):4.0f} || Max {np.max(q_samples):4.0f}')
    print('\nOriginal Dist')
    mean_p, var_p, std_p = np.mean(p_samples), np.var(p_samples), np.std(p_samples)
    min_p, max_p = np.min(p_samples), np.max(p_samples)
    print(f'Mean {mean_p:4.2f} || Var {var_p:4.2f} || Std {std_p:4.2f}'
          f'\nMin: {min_p:4d} || Max {max_p:4d}')
    return q_samples, q_samples_init


def get_for_approx(run_against, categories_n, samples_plot_n):
    if run_against == 'uniform':
        uniform_cats = categories_n
        results_file = f'./Results/mu_xi_unif_{uniform_cats}.pkl'
        p_samples = np.random.randint(size=samples_plot_n, low=0, high=categories_n)
        probs = tf.constant(np.array([1 / uniform_cats for _ in range(uniform_cats)]),
                            dtype=tf.float32, shape=uniform_cats)

    elif run_against == 'uniform_mix':
        initial_point, middle_point, final_point, mass_in_beginning = 0, 10, 25, 0.9
        results_file = f'./Results/mu_xi_unifmix_{categories_n}.pkl'
        p_samples = sample_from_uniform_mix(size=samples_plot_n, initial_point=initial_point,
                                            middle_point=middle_point,
                                            final_point=final_point, mass_in_beginning=mass_in_beginning)
        probs = get_uniform_mix_probs(initial_point=initial_point, middle_point=middle_point,
                                      final_point=final_point, mass_in_beginning=mass_in_beginning,
                                      max_size=categories_n)
        probs = tf.constant(probs, dtype=tf.float32, shape=categories_n)
    elif run_against == 'discrete':
        results_file = f'./Results/mu_xi_discrete.pkl'
        probs = tf.constant(np.array([10., 1., 5., 1., 10., 10., 1., 6., 1., 1.]), dtype=tf.float32)
        probs = probs / np.sum(probs)
        p_samples = np.random.choice(a=probs.shape[0], p=probs.numpy(), size=samples_plot_n)
    elif run_against == 'poisson':
        poisson_mean = 50
        results_file = f'./Results/mu_xi_poisson_{poisson_mean}.pkl'
        poisson_probs = np.array([poisson.pmf(k=k, mu=poisson_mean) for k in range(categories_n)])
        p_samples = np.random.poisson(lam=poisson_mean, size=samples_plot_n)
        probs = tf.constant(poisson_probs, dtype=tf.float32, shape=categories_n)
    elif run_against == 'binomial':
        binomial_p = 0.3
        results_file = f'./Results/mu_xi_binomial_{binomial_p}.pkl'
        binomial_probs = np.array([binom.pmf(k=k, n=categories_n, p=binomial_p)
                                   for k in range(categories_n)])
        p_samples = np.random.binomial(n=categories_n, p=binomial_p, size=samples_plot_n)
        probs = tf.constant(binomial_probs, dtype=tf.float32, shape=categories_n)
    elif run_against == 'geometric':
        geometric_p = 0.4
        results_file = f'./Results/mu_xi_geometric_{geometric_p}.pkl'
        geometric_probs = np.array([geom.pmf(k=k, p=geometric_p) for k in range(categories_n)])
        p_samples = np.random.geometric(p=geometric_p, size=samples_plot_n)
        probs = tf.constant(geometric_probs, dtype=tf.float32, shape=categories_n)
    elif run_against == 'negative_binomial':
        nb_r = 50
        nb_p = 0.6
        results_file = f'./Results/mu_xi_neg_binr{nb_r}.pkl'
        nb_probs = np.array([nbinom.pmf(k=k, n=nb_r, p=nb_p) for k in range(categories_n)])
        p_samples = np.random.negative_binomial(n=nb_r, p=nb_p, size=samples_plot_n)
        probs = tf.constant(nb_probs, dtype=tf.float32, shape=categories_n)
    else:
        raise RuntimeError

    return probs, p_samples, results_file


def plot_loss_and_initial_final_histograms(ax, loss_iter, p_samples, q_samples, q_samples_init,
                                           model_type, y_lim_max, x_lim_max,
                                           title: str, number_of_bins: int = 15):
    total_iterations = loss_iter.shape[0]
    hist_color = '#377eb8'
    ax[0].set_title(title)
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')
    ax[0].plot(np.arange(total_iterations), loss_iter, alpha=0.2)
    window = 100 if total_iterations >= 500 else 10
    loss_df = pd.DataFrame(data=loss_iter).rolling(window=window).mean()
    ax[0].plot(np.arange(total_iterations), loss_df, label=f'mean over {window} iter')
    ax[0].legend()

    ax[1].hist(p_samples, bins=np.arange(number_of_bins),
               color='grey', alpha=0.5, label='p', density=True)
    ax[1].hist(q_samples_init, bins=np.arange(number_of_bins), color=hist_color, alpha=0.5,
               label=model_type, density=True)
    ax[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x: 1.2f}"))
    ax[1].set_ylim([0, y_lim_max])
    ax[1].set_xlim([0, x_lim_max])
    ax[1].set_title('Initial distribution')
    ax[1].legend()

    ax[2].hist(p_samples, bins=np.arange(number_of_bins),
               color='grey', alpha=0.5, label='p', density=True)
    ax[2].hist(q_samples, bins=np.arange(number_of_bins),
               color=hist_color, alpha=0.5, label=model_type, density=True)
    ax[2].set_title('Final distribution')
    ax[2].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x: 1.2f}"))
    ax[2].set_ylim([0, y_lim_max])
    ax[2].set_xlim([0, x_lim_max])
    ax[2].legend()


def plot_initial_final_histograms(ax, p_samples, q_samples, q_samples_init,
                                  model_type, y_lim_max, x_lim_max,
                                  number_of_bins: int = 15):
    hist_color = '#377eb8'
    ax[0].hist(p_samples, bins=np.arange(number_of_bins),
               color='grey', alpha=0.5, label='p', density=True)
    ax[0].hist(q_samples_init, bins=np.arange(number_of_bins), color=hist_color, alpha=0.5,
               label=model_type, density=True)
    ax[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x: 1.2f}"))
    ax[0].set_ylim([0, y_lim_max])
    ax[0].set_xlim([0, x_lim_max])
    ax[0].set_title('Initial distribution')
    ax[0].legend(prop={'size': 12})
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        label.set_fontsize(12)
        label.set_color('black')

    ax[1].hist(p_samples, bins=np.arange(number_of_bins),
               color='grey', alpha=0.5, label='p', density=True)
    ax[1].hist(q_samples, bins=np.arange(number_of_bins),
               color=hist_color, alpha=0.5, label=model_type, density=True)
    ax[1].set_title('Final distribution')
    ax[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x: 1.2f}"))
    ax[1].set_ylim([0, y_lim_max])
    ax[1].set_xlim([0, x_lim_max])
    ax[1].legend(prop={'size': 12})
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        label.set_fontsize(12)
        label.set_color('black')


def plot_histograms_of_gs(ax, p_samples, q_samples_list, q_samples_init_list,
                          y_lim_max, x_lim_max, categories_n_list, number_of_bins: int = 15):
    colors = ['#c5a6fa', '#4e17aa', '#2c0d61']
    ax[0].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p',
               density=True)
    for i in range(len(q_samples_init_list)):
        ax[0].hist(q_samples_init_list[i], bins=np.arange(number_of_bins), color=colors[i], alpha=0.5,
                   label=f'GS with K = {categories_n_list[i]:d}', density=True)
    ax[0].set_ylim([0, y_lim_max])
    ax[0].set_xlim([0, x_lim_max])
    ax[0].set_title('Initial distribution')
    ax[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x: 1.2f}"))
    ax[0].legend(prop={'size': 12})
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        label.set_fontsize(12)
        label.set_color('black')

    ax[1].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p',
               density=True)
    for i in range(len(q_samples_list)):
        ax[1].hist(q_samples_list[i], bins=np.arange(number_of_bins), color=colors[i], alpha=0.5,
                   label=f'GS with K = {categories_n_list[i]:d}', density=True)
    ax[1].set_title('Final distribution')
    ax[1].set_ylim([0, y_lim_max])
    ax[1].set_xlim([0, x_lim_max])
    ax[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x: 1.2f}"))
    ax[1].legend(prop={'size': 12})
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        label.set_fontsize(12)
        label.set_color('black')


def count_zeros_in_gradient(grad_dict):
    grad_np = np.zeros(shape=100)
    i = 0
    for k, v in grad_dict.items():
        v = v.flatten()
        z = np.abs(v) <= 1.e-10
        grad_np[i] = np.mean(z)
        i += 1
    return grad_np


def add_mean_std_plot_line(runs, color, label, offset=5, linestyle='-'):
    shrinked_runs = runs[:, offset:]
    add_mean_lines(shrinked_runs, label=label, color=color, offset=offset, linestyle=linestyle)
    add_std_lines(shrinked_runs, color=color, offset=offset)


def add_mean_lines(runs, color, offset, label, linestyle):
    run_axis = 0
    runs_num = np.arange(runs.shape[1]) + offset
    run_mean = np.mean(runs, axis=run_axis)
    plt.plot(runs_num, run_mean, label=label, color=color, linestyle=linestyle)


def add_std_lines(runs, color, offset, alpha=0.5):
    run_axis = 0
    runs_num = np.arange(runs.shape[1]) + offset
    run_mean = np.mean(runs, axis=run_axis)
    total_runs = runs.shape[run_axis]
    run_std = np.std(runs, axis=run_axis) / total_runs
    plt.vlines(runs_num, ymin=run_mean - run_std, ymax=run_mean + run_std, color=color, alpha=alpha)


def make_np_of_var_from_log_files(variable_name: str, files_list: list, path_to_files: str):
    results_list = []
    for f in files_list:
        if not f.startswith('.'):
            variable_np = get_variable_np_array_from_log_file(variable_name=variable_name,
                                                              path_to_file=path_to_files + f)
            results_list.append(variable_np)
    results_np = create_global_np_array_from_results(results_list=results_list)
    return results_np


def get_variable_np_array_from_log_file(variable_name: str, path_to_file: str):
    variable_results = []
    with open(file=path_to_file, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            split = line.split(sep='||')
            if len(split) > 1:
                for part in split:
                    if part.find(variable_name) > 0:
                        var = float(part.split()[1])
                        variable_results.append(var)
        variable_np = np.array(variable_results)
    return variable_np


def create_global_np_array_from_results(results_list: list):
    total_runs = len(results_list)
    size_of_run = results_list[0].shape[0]
    results_np = np.zeros(shape=(total_runs, size_of_run))
    for run in range(total_runs):
        results_np[run, :] = results_list[run]
    return results_np


def reshape_parameter_for_model(shape, param):
    batch_n, categories_n, sample_size, num_of_vars = shape
    param = np.reshape(param, newshape=(batch_n, categories_n, 1, 1))
    param = np.broadcast_to(param, shape=(batch_n, categories_n, sample_size, 1))
    param = np.broadcast_to(param, shape=(batch_n, categories_n, sample_size, num_of_vars))
    param = tf.constant(param, dtype=tf.float32)
    return param


def convert_into_one_hot(shape, categorical):
    batch_n, categories_n, sample_size, num_of_vars = shape
    categorical_one_hot = np.zeros(shape=shape)
    for i in range(sample_size):
        for j in range(num_of_vars):
            max_i = categorical[0, i, j]
            categorical_one_hot[0, max_i, i, j] = 1.

    return categorical_one_hot


def setup_logger(log_file_name, logger_name: str = None):
    if logger_name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:    %(message)s')
    stream_formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(filename=log_file_name)
    file_handler.setFormatter(fmt=formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=stream_formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=stream_handler)
    logger.propagate = False
    return logger


def append_timestamp_to_file(file_name, termination: str = '.pkl') -> str:
    ending = get_ending_with_timestamp(termination=termination)
    ending_len = len(termination)
    return file_name[:-ending_len] + '_' + ending


def get_ending_with_timestamp(termination: str = '.pkl') -> str:
    current_time = str(datetime.datetime.now())
    parts_of_date = current_time.split(sep=' ')
    year_month_day = parts_of_date[0].replace('-', '')
    hour_min_sec = parts_of_date[1].replace(':', '')
    hour_min = hour_min_sec[:4]
    sec = hour_min_sec[7:10]
    ending = year_month_day + '_' + hour_min + '_' + sec + termination
    return ending


#  Evaluating Simplex Proximity
#  =====================================================================================================

def load_parameter_values(prior_file):
    with open(file=prior_file, mode='rb') as f:
        parameters = pickle.load(f)
    mu = tf.constant(parameters['mu'], dtype=tf.float32)
    xi = tf.constant(parameters['xi'], dtype=tf.float32)
    return mu, xi


def plot_boxplots(model: str, results, temp_grid, mult=5):
    obs_n, samples_n = results[model]['sample'].shape
    rows_list = []
    plt.figure(dpi=150)
    plt.style.use(style='ggplot')
    for i in range(obs_n // mult):
        for s in range(samples_n):
            entry = {'tau': temp_grid[mult * i], 'distance': results[model]['sample'][mult * i, s]}
            rows_list.append(entry)
    df = pd.DataFrame(rows_list)
    ax = sns.boxplot(x='tau', y='distance', data=df, color='royalblue', boxprops={'alpha': 0.5})
    ax.set_xticklabels([f'{tau:0.2f}' for tau in [temp_grid[mult * i]
                                                  for i in range(obs_n // mult)]])
    ax.tick_params(labelrotation=90)
    plt.xlabel('τ')
    plt.ylabel('Distribution of Distance to Simplex Vertex')
    plt.ylim([0., 1.])
    plt.tight_layout()
    plt.savefig(f'./Results/Outputs/{model}_boxplot.png')
    plt.show()


def plot_stat_result(stat: str, models, results, temp_grid):
    colors = ['blue', 'red', 'orange', 'purple', 'green']
    plt.figure(dpi=150)
    plt.title(stat.title() + ' Distance to Simplex Vertex')

    for idx, model in enumerate(models.keys()):
        curve = fit_curve_to_result_along_grid(result=results[model][stat], temp_grid=temp_grid)
        plt.plot(temp_grid, results[model][stat], label=model, color=colors[idx], alpha=0.5)
        plt.plot(temp_grid, curve.predict(temp_grid.reshape(-1, 1)),
                 label=model + ' smooth', color=colors[idx], alpha=0.25, linestyle='--')
    plt.xlabel(r'$\tau$')
    plt.ylabel('Euclidean Distance')
    plt.annotate(s='Using a Uniform Distribution with 50 categories', xy=(1, -40),
                 xycoords='axes points', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Results/Outputs/{stat}_distance_to_simplex.png')
    plt.show()


def fit_curve_to_result_along_grid(result: np.ndarray, temp_grid: np.ndarray, gamma: float = 10.):
    curve = KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma)
    curve.fit(temp_grid.reshape(-1, 1), result.reshape(-1, 1))
    return curve


def run_simulation(samples, temp_grid, models, threshold, stats2run):
    results = create_placeholders_for_statistical_results(stats=stats2run, models=models,
                                                          samples_per_stat=temp_grid.shape[0])
    tic = time.time()
    for model, params in models.items():
        results[model].update({'sample': np.zeros(shape=(temp_grid.shape[0], samples))})
        categories_n = params[0].shape[1] if model == 'GS' else params[0].shape[1] + 1
        results[model].update({'sample_var': np.zeros(shape=(temp_grid.shape[0], categories_n))})
        for i in range(temp_grid.shape[0]):
            temp = tf.constant(temp_grid[i], dtype=tf.float32)
            psi = generate_sample(sample_size=samples, params=params, dist_type=model, temp=temp,
                                  threshold=threshold, output_one_hot=True)[0, :, :, 0]
            diff = calculate_distance_to_simplex(psi=psi, argmax_locs=np.argmax(psi, axis=0))
            results[model]['sample_var'][i, :] = tf.math.reduce_variance(psi, axis=1)
            results[model]['sample'][i, :] = diff[np.argsort(diff)][:samples]
            for stat in stats2run:
                results[model][stat][i] = compute_statistic(stat=stat, samples=diff)
    print(f'\nExperiment took: {time.time() - tic:6.1f} sec')
    return results


@numba.jit(nopython=True, parallel=True)
def calculate_distance_to_simplex(psi, argmax_locs):
    samples_n = psi.shape[1]
    categories_n = psi.shape[0]
    diffs = np.zeros(shape=samples_n)
    for s in numba.prange(samples_n):
        zeros = np.zeros(shape=categories_n)
        zeros[argmax_locs[s]] = 1
        diffs[s] = np.sqrt(np.sum((zeros - psi[:, s]) ** 2))
    return diffs


def compute_statistic(stat, samples):
    if stat[0] == 'p':
        return np.percentile(samples, q=int(stat[1:]))
    else:
        stat2numpy = {'mean': np.mean, 'median': np.median, 'max': np.max, 'min': np.min,
                      'std': np.std}
        return stat2numpy[stat](samples)


def create_placeholders_for_statistical_results(stats, models, samples_per_stat):
    placeholders = {model: {} for model in models.keys()}
    for model in models:
        placeholders[model].update({stat: np.zeros(shape=samples_per_stat) for stat in stats})

    return placeholders


# Initializing Parameters
#  =====================================================================================================

def get_uniform_mix_probs(initial_point: int, middle_point: int, final_point: int, mass_in_beginning,
                          max_size: int) -> np.ndarray:
    probs = np.zeros(shape=max_size)
    points_in_beginning_n = (middle_point - initial_point + 1)
    points_in_end_n = (final_point - middle_point)
    mass_in_end = 1. - mass_in_beginning

    for i in range(final_point - initial_point + 1):
        if i <= middle_point:
            probs[i] = mass_in_beginning / points_in_beginning_n
        else:
            probs[i] = mass_in_end / points_in_end_n
    return probs


def sample_from_uniform_mix(size: int, initial_point: int, middle_point: int, final_point: int,
                            mass_in_beginning):
    first_samples = np.random.randint(low=initial_point, high=middle_point, size=size)
    mixture_samples = np.random.randint(low=middle_point, high=final_point + 1, size=size)
    prob_of_beginning = np.random.uniform(low=0, high=1, size=size)
    samples_in_beginning = np.where(prob_of_beginning > 1. - mass_in_beginning)[0]
    mixture_samples[samples_in_beginning] = first_samples[samples_in_beginning]
    return mixture_samples


def initialize_mu_and_xi_equally(shape, value_mean=0., value_xi=0.):
    mu = tf.constant(value_mean, dtype=tf.float32, shape=shape)
    xi = tf.constant(value_xi, dtype=tf.float32, shape=shape)
    mu = tf.Variable(mu)
    xi = tf.Variable(xi)
    return mu, xi


def initialize_mu_and_xi_for_logistic(shape, seed: int = 21) -> Tuple[tf.Variable, tf.Variable]:
    categories = shape[1]
    np.random.RandomState(seed=seed)
    inv_softplus_one = 0.541324854612918
    mu = np.random.normal(loc=0, scale=0.01, size=shape)
    xi = np.random.normal(loc=inv_softplus_one, scale=0.01, size=shape)
    a = 1 / categories
    if a == 1:
        mu[:, :, :] = 0.
    else:
        mu[:, :categories, :] = tf.math.log(a / (1 - a))

    # noinspection PyArgumentList
    mu = tf.Variable(initial_value=mu, dtype=tf.float32)
    # noinspection PyArgumentList
    xi = tf.Variable(initial_value=xi, dtype=tf.float32)
    return mu, xi

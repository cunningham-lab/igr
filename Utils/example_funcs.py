# ===========================================================================================================
# Documentation
# ===========================================================================================================
"""
Contains the auxiliary functions for the distributions
"""
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Imports
# ===========================================================================================================
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import poisson
from Utils.initializations import get_initial_analytical_mus
from Utils.Distributions import compute_log_logit_normal
from matplotlib import pyplot as plt
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Define the functions
# ===========================================================================================================


def print_latex_table(gs, sb, poisson_means, categories, use_scientific=False):
    # operations = {'Mean': np.mean, 'Std': np.std, 'Median': np.median, 'Min': np.min, 'Max': np.max}
    operations = {'Poi': np.mean}
    for poi_id, poisson_mean in enumerate(poisson_means):
        for oper_name, oper in operations.items():
            s = '& '
            for cat_id, categories_n in enumerate(categories):
                aux = gs[cat_id, poi_id, :]
                if use_scientific:
                    s += f'{oper(aux):2.2e} & '
                else:
                    s += f'{oper(aux):4.2f} & '

            if use_scientific:
                s += f'{oper(sb[poi_id, :]):2.2e} '
            else:
                s += f'{oper(sb[poi_id, :]):4.2f} '
            if oper_name == 'Max':
                ob = '\multicolumn{1}{|l|}{' + oper_name + '(' + str(poisson_mean) + ')}'
                print(ob)
                print(s + '\ \ \hline')
            else:
                ob = '\multicolumn{1}{|l|}{' + oper_name + '(' + str(poisson_mean) + ')}'
                print(ob)
                print(s + '\ \ ')


def plot_experiment_results(gs, sb, poisson_means, categories, y_label, analysis_type, ylim, logit=None):
    fig, ax = plt.subplots(nrows=1, ncols=len(poisson_means), figsize=(15, 5), dpi=200)
    for poi_idx, poisson_mean in enumerate(poisson_means):
        results_dist = []
        ticks = []
        for cat_id, categories_n in enumerate(categories):
            results_dist.append(gs[cat_id, poi_idx, :])
            ticks.append(f'GS({categories_n})')
        results_dist.append(sb[poi_idx, :])
        ticks.append('SB')
        if logit is not None:
            results_dist.append(logit[poi_idx, :])
            ticks.append('Logit')
        ax[poi_idx].boxplot(results_dist, positions=np.array([i + 1 for i in range(len(results_dist))]))
        ax[poi_idx].set_xticklabels(ticks)
        ax[poi_idx].set_title(f'Poisson Mean {poisson_mean}')
        ax[poi_idx].set_ylabel(y_label)
        ax[poi_idx].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(fname=f'./Results/{analysis_type}_plot.png')
    plt.show()


def plot_loss_and_initial_final_histograms(ax, loss_iter, p_samples, q_samples, q_samples_init,
                                           title: str, number_of_bins: int = 15):
    total_iterations = loss_iter.shape[0]
    ylim = 0.2
    ax[0].set_title(title)
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')
    ax[0].plot(np.arange(total_iterations), loss_iter, alpha=0.2)
    window = 100 if total_iterations >= 500 else 10
    loss_df = pd.DataFrame(data=loss_iter).rolling(window=window).mean()
    ax[0].plot(np.arange(total_iterations), loss_df, label=f'mean over {window} iter')
    ax[0].legend()

    ax[1].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p', density=True)
    ax[1].hist(q_samples_init, bins=np.arange(number_of_bins), color='red', alpha=0.5,
               label='q', density=True)
    ax[1].set_ylim([0, ylim])
    ax[1].set_title('Initial distribution')
    ax[1].set_ylabel('Normalized Counts')
    ax[1].set_xlabel('Bins')
    ax[1].legend()

    ax[2].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p', density=True)
    ax[2].hist(q_samples, bins=np.arange(number_of_bins), color='red', alpha=0.5, label='q', density=True)
    ax[2].set_title('Final distribution')
    ax[2].set_ylim([0, ylim])
    ax[2].set_ylabel('Normalized Counts')
    ax[2].set_xlabel('Bins')
    ax[2].legend()


def plot_histograms_of_gs(ax, p_samples, q_samples_list, q_samples_init_list, number_of_bins: int = 15):
    colors = ['green', 'blue', 'orange']
    k = [20, 40, 100]
    # y_lim = 0.35
    # k = [10]
    y_lim = 0.2
    x_lim = 60
    ax[0].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p',
               density=True)
    for i in range(len(q_samples_init_list)):
        ax[0].hist(q_samples_init_list[i], bins=np.arange(number_of_bins), color=colors[i], alpha=0.5,
                   label=f'GS with K = {k[i]:d}', density=True)
    ax[0].set_ylim([0, y_lim])
    ax[0].set_xlim([0, x_lim])
    ax[0].set_title('Initial distribution')
    # ax[0].set_ylabel('Normalized Counts')
    ax[0].legend()

    ax[1].hist(p_samples, bins=np.arange(number_of_bins), color='grey', alpha=0.5, label='p',
               density=True)
    for i in range(len(q_samples_list)):
        ax[1].hist(q_samples_list[i], bins=np.arange(number_of_bins), color=colors[i], alpha=0.5,
                   label=f'GS with K = {k[i]:d}', density=True)
    ax[1].set_title('Final distribution')
    ax[1].set_ylim([0, y_lim])
    ax[1].set_xlim([0, x_lim])
    # ax[1].set_ylabel('Normalized Counts')
    ax[1].legend()


def compute_poisson_continuous(psi: tf.Tensor, mean: tf.Tensor) -> tf.Tensor:
    k = compute_num(psi=psi)
    # noinspection PyTypeChecker
    norm_cons = 1 / tf.math.exp(tf.math.lgamma(k + 1))
    poi = tf.math.exp(-mean) * norm_cons * mean ** k
    return poi


def compute_log_poisson_cont(psi: tf.Tensor, mean: tf.Tensor) -> tf.Tensor:
    k = compute_num(psi=psi)
    log_poi = -mean + k * tf.math.log(mean) - tf.math.lgamma(tf.constant(1, dtype=tf.float32) + k)
    return log_poi


def compute_num(psi: tf.Tensor) -> tf.Tensor:
    size_n = psi.shape[0]
    running = tf.constant(value=np.arange(size_n), dtype=tf.float32, shape=(size_n, 1))
    expectation = tf.reduce_sum(running * psi, axis=0)
    return expectation


def compute_log_q_poisson(kappa: tf.Tensor,
                          delta: tf.Tensor,
                          n_required: int,
                          poisson_mean: tf.Tensor) -> tf.Tensor:
    probs = np.array([poisson.pmf(k=k, mu=poisson_mean.numpy()) for k in range(n_required)])
    mu_0 = np.zeros(shape=(n_required, kappa.shape[1]))
    mu_0[:n_required, :] = get_initial_analytical_mus(probs=probs, initialization_type='stick-break')
    mu_0 = tf.constant(value=mu_0, dtype=tf.float32)
    chi_0 = tf.constant(value=0.541324854612918, dtype=tf.float32,
                        shape=(n_required, kappa.shape[1]))
    sigma_0 = tf.math.softplus(chi_0)
    epsilon_0 = (delta[:n_required] - mu_0) / sigma_0
    log_q_kappa = compute_log_logit_normal(kappa=kappa[:n_required],
                                           sigma=sigma_0[:n_required],
                                           epsilon=epsilon_0[:n_required])
    return log_q_kappa


def sample_poisson_truncated(truncated_at: int, sample_size: int, poisson_mean: np.ndarray) -> np.ndarray:
    gathered, iteration = 0, 0
    max_iter = 1000
    poisson_truncated_sample = np.zeros(shape=sample_size)
    while (gathered < sample_size) & (iteration < max_iter):
        poi_sample = np.random.poisson(lam=poisson_mean, size=1)
        if poi_sample <= truncated_at:
            poisson_truncated_sample[gathered] = poi_sample
            gathered += 1
        iteration += 1
    return poisson_truncated_sample
# ===========================================================================================================

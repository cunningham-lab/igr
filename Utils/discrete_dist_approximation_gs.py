# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Import Packages & Set parameters
# ===========================================================================================================
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import geom
from scipy.stats import nbinom
from Utils.MinimizeEmpiricalLoss import MinimizeEmpiricalLoss, get_initial_params_for_model_type
from Utils.Distributions import generate_sample
# from Utils.example_funcs import plot_loss_and_initial_final_histograms
from Utils.example_funcs import plot_histograms_of_gs
from Utils.initializations import get_uniform_mix_probs, sample_from_uniform_mix

sample_size = 1000
total_iterations = 2 * int(1.e2)
learning_rate = 1.e-2

# Global parameters
samples_plot_n = int(1.e4)
batch_n = 1
np.random.RandomState(seed=21)
pool = Pool()
temp_init, temp_final = 0.1, 0.1
threshold = 0.99
# categories_n = 10
# categories_n = 12
categories_n = 100
# categories_n = 200
shape = (batch_n, categories_n, 1, 1)
type_temp_schedule = 'constant'
model_type = 'ExpGS'
# model_type = 'sb'
# model_type = 'GauSoftMax'

skip_first_iterations = 10
tolerance = 1.e-5

uniform_probs = np.array([1 / categories_n for _ in range(categories_n)])

run_against = 'negative_binomial'
# run_against = 'poisson'
# run_against = 'discrete'
# run_against = 'binomial'

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
    categories_n = probs.shape[0]
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
    binomial_probs = np.array([binom.pmf(k=k, n=categories_n, p=binomial_p) for k in range(categories_n)])
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

mean_p, var_p, std_p = np.mean(p_samples), np.var(p_samples), np.std(p_samples)
min_p, max_p = np.min(p_samples), np.max(p_samples)
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Train model
# ===========================================================================================================
q_samples_list = []
q_samples_init_list = []
categories_n_list = [20, 40, 100]
# categories_n_list = [10]
for i in range(len(categories_n_list)):
    categories_n = categories_n_list[i]
    shape = (batch_n, categories_n, 1, 1)
    params, params_init = get_initial_params_for_model_type(model_type=model_type, shape=shape)

    minimizer = MinimizeEmpiricalLoss(learning_rate=learning_rate, temp_init=temp_init,
                                      temp_final=temp_final, pool=pool, sample_size=sample_size,
                                      tolerance=tolerance, run_kl=True,
                                      max_iterations=total_iterations, model_type=model_type,
                                      type_temp_schedule=type_temp_schedule)
    minimizer.set_variables(params=params)
    minimizer.threshold = threshold
    minimizer.run_iteratively = True
    minimizer.optimize_model(mean_p=mean_p, var_p=var_p, probs=probs, p_samples=p_samples)

    temp = tf.constant(0.1, dtype=tf.float32)
    q_samples = np.zeros(shape=samples_plot_n)
    q_samples_init = np.zeros(shape=samples_plot_n)
    for sample_id in range(samples_plot_n):
        q_samples[sample_id] = generate_sample(sample_size=1, params=params, temp=temp,
                                               threshold=minimizer.threshold,
                                               dist_type=model_type)[0, 0]
        q_samples_init[sample_id] = generate_sample(sample_size=1, params=params_init, dist_type=model_type,
                                                    temp=temp, threshold=minimizer.threshold)[0, 0]
    categories_n
    q_samples_list.append(q_samples)
    q_samples_init_list.append(q_samples_init)
    print(f'{model_type}')
    print(f'Mean {np.mean(minimizer.q_samples):4.2f} || '
          f'Var {np.var(minimizer.q_samples):4.2f} || '
          f'Std {np.std(minimizer.q_samples):4.2f}'
          f'\nMin: {np.min(minimizer.q_samples):4.0f} || Max {np.max(minimizer.q_samples):4.0f}')
    print('\nOriginal Dist')
    print(f'Mean {mean_p:4.2f} || Var {var_p:4.2f} || Std {std_p:4.2f}'
          f'\nMin: {min_p:4d} || Max {max_p:4d}')
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Plot Loss and Histograms
# ===========================================================================================================
plt.style.use(style='ggplot')
# q_samples_list = [q_samples]
# q_samples_init_list = [q_samples_init]
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
plot_histograms_of_gs(ax=ax, p_samples=p_samples, q_samples_list=q_samples_list,
                      q_samples_init_list=q_samples_init_list, number_of_bins=categories_n + 2)
plt.tight_layout()
plt.savefig(fname='./Results/gs_hist.png')
plt.show()
# ===========================================================================================================
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
plot_histograms_of_gs(ax=ax, p_samples=p_samples, q_samples_list=q_samples_list,
                      q_samples_init_list=q_samples_init_list, number_of_bins=categories_n + 2)
plt.tight_layout()
plt.savefig(fname='./Results/gs_hist.png')
plt.show()
# ===========================================================================================================

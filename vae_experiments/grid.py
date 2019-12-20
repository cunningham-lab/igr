# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Load Imports and  data
# ===========================================================================================================
import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = True
# model_type = 'ExpGS'
model_type = 'GSM'
# model_type = 'SB'

hyper = {'dataset_name': 'mnist', 'sample_size': 1, 'n_required': 10, 'num_of_discrete_var': 1,
         'latent_norm_n': 10, 'num_of_norm_var': 1, 'num_of_norm_param': 2,
         'architecture': 'conv_jointvae', 'learning_rate': 0.0005, 'batch_n': 64, 'epochs': 100,
         'use_analytical_in_test': True, 'run_analytical_kl': True,
         'prior_file': './Results/mu_xi_unif_10_ng.pkl',
         'run_ccβvae': True, 'γ': tf.constant(30.), 'num_of_discrete_param': 2,
         'continuous_c_linspace': (5., 5., 1), 'discrete_c_linspace': (5., 5., 1)}
hyper.update({'latent_discrete_n': hyper['n_required']})

experiment = {
    1: {'model_type': 'GSM', 'temp': 0.5, 'γ': tf.constant(30.),
        'discrete_c_linspace': (5., 5., 1), 'continuous_c_linspace': (5., 5., 1)},

    2: {'model_type': 'GSM', 'temp': 0.5, 'γ': tf.constant(30.),
        'discrete_c_linspace': (0., 0., 1), 'continuous_c_linspace': (5., 5., 1)},

    7: {'model_type': 'GSM', 'temp': 0.5, 'γ': tf.constant(30.),
        'discrete_c_linspace': (5., 5., 1), 'continuous_c_linspace': (10., 10., 1)},

    3: {'model_type': 'GSM', 'temp': 0.1, 'γ': tf.constant(30.),
        'discrete_c_linspace': (5., 5., 1), 'continuous_c_linspace': (10., 10., 1)},

    9: {'model_type': 'GSM', 'temp': 0.1, 'γ': tf.constant(100.),
        'discrete_c_linspace': (5., 5., 1), 'continuous_c_linspace': (30., 30., 1)},

    8: {'model_type': 'GSM', 'temp': 0.5, 'γ': tf.constant(100.),
        'discrete_c_linspace': (5., 5., 1), 'continuous_c_linspace': (30., 30., 1)},

    4: {'model_type': 'GSM', 'temp': 0.1, 'γ': tf.constant(100.),
        'discrete_c_linspace': (5., 5., 1), 'continuous_c_linspace': (5., 5., 1)},

    5: {'model_type': 'GSM', 'temp': 0.1, 'γ': tf.constant(100.),
        'discrete_c_linspace': (10., 10., 1), 'continuous_c_linspace': (5., 5., 1)}}

for _, d in experiment.items():
    for key, value in d.items():
        hyper[key] = value
    run_vae(hyper=hyper, run_with_sample=run_with_sample)
# ===========================================================================================================

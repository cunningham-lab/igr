# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Load Imports and  data
# ===========================================================================================================
import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = False
model_type = 'GSPDis'
hyper = {'dataset_name': 'mnist', 'sample_size': 1, 'n_required': 10, 'num_of_discrete_var': 30,
         'latent_norm_n': 0, 'num_of_norm_var': 0, 'num_of_norm_param': 0,
         'learning_rate': 0.001, 'batch_n': 64, 'epochs': 100, 'architecture': 'dense',
         'prior_file': './Results/mu_xi_unif_10_ng.pkl',
         'run_ccβvae': False, 'γ': tf.constant(30.),
         'continuous_c_linspace': (0., 5., 25_000), 'discrete_c_linspace': (0., 5., 25_000)}
hyper.update({'model_type': model_type, 'temp': 0.10, 'num_of_discrete_param': 2,
              'use_analytical_in_test': True, 'run_analytical_kl': True})
hyper.update({'latent_discrete_n': hyper['n_required']})

experiment = {1: {'model_type': 'ExpGSDis', 'temp': 0.67, 'run_analytical_kl': False,
                  'num_of_discrete_param': 1},
              2: {'model_type': 'ExpGSDis', 'temp': 0.15, 'run_analytical_kl': False,
                  'num_of_discrete_param': 1},
              3: {'model_type': 'GSMDis', 'temp': 0.50, 'run_analytical_kl': True,
                  'num_of_discrete_param': 2},
              4: {'model_type': 'GSMDis', 'temp': 0.10, 'run_analytical_kl': True,
                  'num_of_discrete_param': 2}}

for _, d in experiment.items():
    for key, value in d.items():
        hyper[key] = value
    run_vae(hyper=hyper, run_with_sample=run_with_sample)
# ===========================================================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Load Imports and  data
# ===========================================================================================================
import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = True
model_type = 'ExpGSDis'
# model_type = 'GSMDis'

hyper = {'dataset_name': 'omniglot', 'sample_size': 1, 'n_required': 10,
         'latent_norm_n': 0, 'num_of_discrete_var': 30, 'num_of_norm_var': 0, 'num_of_norm_param': 0,
         'learning_rate': 0.001, 'batch_n': 64, 'epochs': 100, 'architecture': 'dense',
         'run_ccβvae': False, 'γ': tf.constant(30.),
         'continuous_c_linspace': (0, 5, 25_000), 'discrete_c_linspace': (0, 5, 25_000)}

if model_type in ['GS', 'ExpGS', 'GSDis', 'ExpGSDis']:
    hyper.update({'model_type': model_type, 'temp': 0.67, 'num_of_discrete_param': 1,
                  'use_analytical_in_test': False, 'run_analytical_kl': False})
    hyper.update({'temp_min': hyper['temp']})
    hyper.update({'latent_discrete_n': hyper['n_required']})

elif model_type in ['GSM', 'GSP', 'GSMDis', 'IGSMDis', 'GSPDis']:
    hyper.update({'model_type': model_type, 'temp': 0.50,
                  'prior_file': './Results/mu_xi_unif_10_ng.pkl', 'num_of_discrete_param': 2,
                  'use_analytical_in_test': True, 'run_analytical_kl': True})
    hyper.update({'temp_min': hyper['temp']})
    hyper.update({'latent_discrete_n': hyper['n_required']})

elif model_type == 'SB':
    hyper.update({'latent_discrete_n': 50, 'threshold': 0.99, 'latent_norm_n': 10, 'temp': 0.1,
                  'model_type': model_type, 'num_of_discrete_param': 2,
                  'prior_file': './Results/mu_xi_unif_50.pkl', 'n_required': 10,
                  'use_analytical_in_test': True, 'run_analytical_kl': True})
    hyper.update({'temp_min': hyper['temp']})
else:
    raise RuntimeError
run_vae(hyper=hyper, run_with_sample=run_with_sample)

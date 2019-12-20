# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Load Imports and  data
# ===========================================================================================================
import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = False
# model_type = 'GSM'
# model_type = 'ExpGS'
model_type = 'SB'

hyper = {'dataset_name': 'mnist', 'sample_size': 1, 'n_required': 10, 'num_of_discrete_var': 1,
         'latent_norm_n': 10, 'num_of_norm_var': 1, 'num_of_norm_param': 2,
         'architecture': 'conv_jointvae', 'learning_rate': 0.0005, 'batch_n': 64, 'epochs': 100,
         'run_ccβvae': True, 'γ': tf.constant(30.),
         'continuous_c_linspace': (0., 5., 25_000), 'discrete_c_linspace': (0., 5., 25_000)}
hyper.update({'latent_discrete_n': hyper['n_required']})

if model_type in ['GS', 'ExpGS', 'GSDis', 'ExpGSDis']:
    hyper.update({'model_type': model_type, 'temp': 0.15, 'num_of_discrete_param': 1,
                  'use_analytical_in_test': False, 'run_analytical_kl': False})

elif model_type in ['GSM', 'GSP', 'GSMDis', 'IGSMDis', 'GSPDis']:
    hyper.update({'model_type': model_type, 'temp': 0.10,
                  'prior_file': './Results/mu_xi_unif_10_ng.pkl', 'num_of_discrete_param': 2,
                  'use_analytical_in_test': True, 'run_analytical_kl': True})

elif model_type == 'SB':
    hyper.update({'latent_discrete_n': 50, 'threshold': 0.99, 'latent_norm_n': 10, 'temp': 0.1,
                  'model_type': model_type, 'num_of_discrete_param': 2, 'truncation_option': 'quantile',
                  'prior_file': './Results/mu_xi_unif_50_sb.pkl', 'n_required': 10,
                  'use_analytical_in_test': True, 'run_analytical_kl': True})
else:
    raise RuntimeError
run_vae(hyper=hyper, run_with_sample=run_with_sample)
# ===========================================================================================================

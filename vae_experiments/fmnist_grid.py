import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = True
# model_type = 'GSM'
# model_type = 'ExpGS'
model_type = 'SB'

hyper = {'dataset_name': 'fmnist', 'sample_size': 1, 'n_required': 10, 'num_of_discrete_var': 1,
         'latent_norm_n': 10, 'num_of_norm_var': 1, 'num_of_norm_param': 2,
         'architecture': 'conv_jointvae', 'learning_rate': 0.0005, 'batch_n': 64, 'epochs': 100,
         'run_ccβvae': True, 'γ': tf.constant(30.),
         'continuous_c_linspace': (0., 5., 25_000), 'discrete_c_linspace': (0., 5., 25_000)}
hyper.update({'latent_discrete_n': hyper['n_required']})


experiment = {1: {'model_type': 'ExpGS', 'temp': 0.15, 'run_analytical_kl': False,
                  'num_of_discrete_param': 1, 'use_analytical_in_test': False},
              2: {'model_type': 'GSM', 'temp': 0.10, 'run_analytical_kl': True,
                  'num_of_discrete_param': 2, 'use_analytical_in_test': True,
                  'prior_file': './Results/mu_xi_unif_10_ng.pkl'},
              3: {'latent_discrete_n': 50, 'threshold': 0.99, 'temp': 0.1,
                  'model_type': 'SB', 'num_of_discrete_param': 2, 'truncation_option': 'quantile',
                  'prior_file': './Results/mu_xi_unif_50_sb.pkl',
                  'use_analytical_in_test': True, 'run_analytical_kl': True}
              }

for _, d in experiment.items():
    for key, value in d.items():
        hyper[key] = value
    run_vae(hyper=hyper, run_with_sample=run_with_sample)

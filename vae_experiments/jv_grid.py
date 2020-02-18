import tensorflow as tf
from Models.train_vae import run_vae_for_all_cases

run_with_sample = True
num_of_repetitions = 3
temps = [0.1]
# temps = [0.01, 0.03, 0.07, 0.10, 0.15, 0.25, 0.50, 0.67]
# temps = [0.10, 0.30, 0.50, 0.67, 0.85, 1.00, 1.11, 1.25]
model_cases = {
    1: {'model_type': 'IGR_I', 'n_required': 9},
    # 2: {'model_type': 'IGR_Planar', 'n_required': 9},
    # 3: {'model_type': 'IGR_SB_Finite', 'n_required': 9,
    #     'prior_file': './Results/mu_xi_unif_10_IGR_SB_Finite.pkl'},
    # 4: {'model_type': 'IGR_SB', 'n_required': 49,
    #     'prior_file': './Results/mu_xi_unif_50_IGR_SB_Finite.pkl',
    #     'threshold': 0.9, 'truncation_option': 'quantile'},
    # 5: {'model_type': 'GS', 'n_required': 10},
}
dataset_cases = {
    1: {'dataset_name': 'mnist', 'gamma': tf.constant(30.), 'latent_norm_n': 10,
        'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)},
    2: {'dataset_name': 'fmnist', 'gamma': tf.constant(30.), 'latent_norm_n': 10,
        'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)},
    # 3: {'dataset_name': 'celeb_a', 'gamma': tf.constant(100.), 'latent_norm_n': 32,
    #     'cont_c_linspace': (0., 50., 100_000), 'disc_c_linspace': (0., 10., 100_000)}
}
hyper = {'num_of_norm_param': 2, 'num_of_norm_var': 1, 'num_of_discrete_var': 1,
         'learning_rate': 0.001, 'batch_n': 64, 'epochs': 50, 'sample_size': 1,
         'run_jv': True, 'architecture': 'conv_jointvae', 'check_every': 10}

run_vae_for_all_cases(hyper, model_cases, dataset_cases, temps, num_of_repetitions, run_with_sample)

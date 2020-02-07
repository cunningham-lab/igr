import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = True
# model_type = 'GS_Dis'
# model_type = 'IGR_I_Dis'
# model_type = 'IGR_SB_Finite_Dis'
model_type = 'IGR_SB_Dis'
# model_type = 'IGR_Planar_Dis'

hyper = {'dataset_name': 'mnist', 'sample_size': 1, 'n_required': 9, 'num_of_discrete_var': 30,
         'latent_norm_n': 0, 'num_of_norm_var': 0, 'num_of_norm_param': 0,
         'learning_rate': 0.001, 'batch_n': 64, 'epochs': 100, 'architecture': 'dense',
         'run_jv': False, 'γ': tf.constant(30.), 'check_every': 10,
         'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)}
hyper.update({'latent_discrete_n': hyper['n_required'] + 1})

if model_type in ['GS', 'GS_Dis']:
    hyper.update({'model_type': model_type, 'temp': 0.25, 'num_of_discrete_param': 1,
                  'run_closed_form_kl': False, 'n_required': hyper['n_required'] + 1})

elif model_type in ['IGR_I_Dis', 'IGR_SB_Dis', 'IGR_SB_Finite_Dis', 'IGR_Planar_Dis']:
    hyper.update({'model_type': model_type, 'temp': 0.30, 'threshold': 0.99, 'truncation_option': 'quantile',
                  'prior_file': './Results/mu_xi_unif_10_IGR_SB_Finite.pkl', 'num_of_discrete_param': 2,
                  'run_closed_form_kl': True})
else:
    raise RuntimeError
run_vae(hyper=hyper, run_with_sample=run_with_sample)

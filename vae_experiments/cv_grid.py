import tensorflow as tf
from Models.train_vae import run_vae

run_with_sample = True
num_of_repetitions = 1
temps = [0.1]
# temps = [0.01, 0.03, 0.07, 0.10, 0.15, 0.25, 0.50, 0.67]
# temps = [0.10, 0.30, 0.50, 0.67, 0.85, 1.00, 1.11, 1.25]
model_cases = {
    1: {'model_type': 'IGR_I_Dis', 'n_required': 9},
    # 2: {'model_type': 'IGR_Planar_Dis', 'n_required': 9},
    # 3: {'model_type': 'IGR_SB_Finite_Dis', 'n_required': 9,
    #     'prior_file': './Results/mu_xi_unif_10_IGR_SB_Finite.pkl'},
    # 4: {'model_type': 'IGR_SB_Dis', 'n_required': 49,
    #     'prior_file': './Results/mu_xi_unif_50_IGR_SB_Finite.pkl',
    #     'threshold': 0.9, 'truncation_option': 'quantile'},
    # 5: {'model_type': 'GS_Dis', 'n_required': 10},
}
dataset_cases = {
    1: {'dataset_name': 'mnist', 'architecture': 'dense'},
    # 2: {'dataset_name': 'fmnist', 'architecture': 'dense'},
    # 3: {'dataset_name': 'celeb_a', 'architecture': 'conv_jointvae'},
}
for _, mod in model_cases.items():
    hyper = {'latent_norm_n': 0, 'num_of_norm_param': 0, 'num_of_norm_var': 0,
             'num_of_discrete_var': 30,
             'learning_rate': 0.001, 'batch_n': 64, 'epochs': 100, 'sample_size': 1,
             'run_jv': False, 'γ': tf.constant(30.), 'check_every': 10,
             'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)}
    i = 0
    experiment = {}
    for k, v in mod.items():
        hyper[k] = v

    hyper['latent_discrete_n'] = hyper['n_required']
    if hyper['model_type'].find('GS') >= 0:
        hyper['run_closed_form_kl'] = False
        hyper['num_of_discrete_param'] = 1
    else:
        hyper['latent_discrete_n'] += 1
        hyper['run_closed_form_kl'] = True
        hyper['num_of_discrete_param'] = 2
    for _, c in dataset_cases.items():
        for t in temps:
            i += 1
            experiment.update({i: {}})
            c.update({'temp': t})
            for key, val in c.items():
                experiment[i][key] = val

    for _, d in experiment.items():
        for key, value in d.items():
            hyper[key] = value
        for rep in range(num_of_repetitions):
            run_vae(hyper=hyper, run_with_sample=run_with_sample)

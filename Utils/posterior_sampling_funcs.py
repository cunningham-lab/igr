import pickle
import numpy as np
from Utils.general import calculate_distance_to_simplex
from Models.train_vae import construct_nets_and_optimizer
from Utils.Distributions import IGR_I, GS, IGR_SB_Finite, IGR_Planar
from Utils.load_data import load_vae_dataset


# Posterior Sampling Funcs
#  ====================================================================================================================
def sample_from_posterior(path_to_results, hyper_file, dataset_name, weights_file,
                          model_type, run_with_sample=True):
    with open(file=path_to_results + hyper_file, mode='rb') as f:
        hyper = pickle.load(f)

    data = load_vae_dataset(dataset_name=dataset_name, batch_n=hyper['batch_n'], epochs=hyper['epochs'],
                            run_with_sample=run_with_sample, architecture=hyper['architecture'], hyper=hyper)
    train_dataset, test_dataset, np_test_images, hyper = data

    vae_opt = construct_nets_and_optimizer(hyper=hyper, model_type=model_type)
    vae_opt.nets.load_weights(filepath=path_to_results + weights_file)

    samples_n, total_test_images, im_idx = 100, 10_000, 0
    # samples_n, total_test_images, im_idx = 100, 19_962, 0
    shape = (total_test_images, samples_n, hyper['num_of_discrete_var'])
    diff = np.zeros(shape=shape)

    for test_image in test_dataset:
        z, x_logit, params = vae_opt.perform_fwd_pass(test_image)
        if model_type.find('Planar') > 0:
            dist = determine_distribution(model_type=model_type, params=params, temp=hyper['temp'],
                                          samples_n=samples_n, planar_flow=vae_opt.nets.planar_flow)
        else:
            dist = determine_distribution(model_type=model_type, params=params, temp=hyper['temp'],
                                          samples_n=samples_n)
        dist.generate_sample()
        psi = dist.psi.numpy()
        for i in range(psi.shape[0]):
            for k in range(psi.shape[3]):
                diff[im_idx, :, k] = calculate_distance_to_simplex(
                    psi=psi[i, :, :, k], argmax_locs=np.argmax(psi[i, :, :, k], axis=0))
            im_idx += 1
    return diff


def determine_distribution(model_type, params, temp, samples_n, planar_flow=None):
    if model_type == 'IGR_I_Dis':
        dist = IGR_I(mu=params[0], xi=params[1], sample_size=samples_n, temp=temp)
    elif model_type == 'IGR_Planar_Dis':
        dist = IGR_Planar(mu=params[0], xi=params[1], sample_size=samples_n, temp=temp,
                          planar_flow=planar_flow)
    elif model_type == 'GS_Dis':
        dist = GS(log_pi=params[0], sample_size=samples_n, temp=temp)
    elif model_type == 'IGR_SB_Finite_Dis':
        dist = IGR_SB_Finite(mu=params[0], xi=params[1], sample_size=samples_n, temp=temp)
    else:
        raise RuntimeError
    return dist

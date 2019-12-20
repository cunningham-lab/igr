import pickle
import numpy as np
from Utils.Distributions import ExpGSDist
from Utils.Distributions import GaussianSoftmaxDist
from Models.VAENet import setup_model
from Models.train_vae import setup_vae_optimizer
from GS_vs_SB_analysis.simplex_proximity_funcs import calculate_distance_to_simplex
from Utils.load_data import load_vae_dataset


def sample_from_posterior(path_to_results, hyper_file, dataset_name, weights_file, model_type, run_with_sample=True):
    with open(file=path_to_results + hyper_file, mode='rb') as f:
        hyper = pickle.load(f)

    data = load_vae_dataset(dataset_name=dataset_name, batch_n=hyper['batch_n'], epochs=hyper['epochs'],
                            run_with_sample=run_with_sample, architecture=hyper['architecture'])
    (train_dataset, test_dataset, test_images, hyper['batch_n'], hyper['epochs'],
     image_size, hyper['iter_per_epoch']) = data

    model = setup_model(hyper=hyper, image_size=image_size)
    model.load_weights(filepath=path_to_results + weights_file)
    vae_opt = setup_vae_optimizer(model=model, hyper=hyper, model_type=model_type)

    samples_n, total_test_images, im_idx = 100, 10_000, 0
    shape = (total_test_images, samples_n, hyper['num_of_discrete_var'])
    diff = np.zeros(shape=shape)

    for test_image in test_dataset:
        z, x_logit, params = vae_opt.perform_fwd_pass(test_image)
        dist = determine_distribution(model_type=model_type, params=params, temp=hyper['temp'], samples_n=samples_n)
        dist.do_reparameterization_trick()
        ψ = dist.psi.numpy()
        for i in range(ψ.shape[0]):
            for k in range(ψ.shape[3]):
                diff[im_idx, :, k] = calculate_distance_to_simplex(
                    ψ=ψ[i, :, :, k], argmax_locs=np.argmax(ψ[i, :, :, k], axis=0))
            im_idx += 1
    return diff


def determine_distribution(model_type, params, temp, samples_n):
    if model_type == 'GSMDis':
        dist = GaussianSoftmaxDist(mu=params[0], xi=params[1], sample_size=samples_n, temp=temp)
    elif model_type == 'ExpGSDis':
        dist = ExpGSDist(log_pi=params[0], sample_size=samples_n, temp=temp)
    else:
        raise RuntimeError
    return dist

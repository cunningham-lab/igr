import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import plotly.graph_objs as go
import plotly.offline as py
import tensorflow as tf
from scipy.stats import norm


def plot_reconstructions_samples_and_traversals(model, hyper, epoch, results_path, test_images, vae_opt):
    if hyper['latent_norm_n'] > 0:
        plot_grid_of_fixed_cont_traversal_along_all_disc_dim(
            model=model, fixed_cont_dim=np.random.choice(a=hyper['latent_norm_n'], size=1)[0],
            discrete_dim_n=hyper['latent_discrete_n'],
            total_discrete_n_to_traverse=vae_opt.n_required,
            cont_dim_n=hyper['latent_norm_n'], traversal_n=12,
            plots_path=results_path + f'/sample_{epoch}_')

    plot_grid_of_random_cont_samples_along_all_disc_dim(
        model=model, samples_n=12,
        cont_dim_n=hyper['latent_norm_n'],
        discrete_dim_n=hyper['latent_discrete_n'] * hyper['num_of_discrete_var'],
        plots_path=results_path + f'/sample_{epoch}_',
        total_discrete_n_to_traverse=vae_opt.n_required)

    _, x_logit, *_ = vae_opt.perform_fwd_pass(x=test_images.astype(np.float32))
    if hyper['dataset_name'] == 'celeb_a' or hyper['dataset_name'] == 'fmnist':
        recon_probs = tf.math.sigmoid(x_logit[0])
        # recon_probs = tf.math.sigmoid(x_logit)
    else:
        recon_probs = tf.math.sigmoid(x_logit)
    plt.figure(figsize=(5, 4), dpi=100)
    for i in range(test_images.shape[0]):
        plt.subplot(5, 4, i + 1)
        plot_based_on_color_or_black(recon_image=recon_probs[i, :, :, 0])
        plt.axis('off')
    plt.savefig(results_path + f'/Reconstruction_{epoch}.png')


def plot_originals(test_images, results_path):
    plt.figure(figsize=(5, 4), dpi=100)
    for i in range(test_images.shape[0]):
        plt.subplot(5, 4, i + 1)
        plot_based_on_color_or_black(recon_image=test_images[i, :, :])
        plt.axis('off')
    plt.savefig(results_path + f'/Reconstruction.png')


def plot_based_on_color_or_black(recon_image, rgb_location=2):
    recon_image.shape
    if recon_image.shape[rgb_location] > 1:
        plt.imshow(recon_image)
    else:
        plt.imshow(recon_image[:, :, 0], cmap='gray')


def plot_grid_of_fixed_cont_traversal_along_all_disc_dim(model, fixed_cont_dim, discrete_dim_n,
                                                         cont_dim_n, traversal_n,
                                                         total_discrete_n_to_traverse,
                                                         plots_path):
    all_latent_samples = traverse_all_discrete_dim_with_fixed_cont_dim(
        fixed_cont_dim=fixed_cont_dim, cont_dim_n=cont_dim_n, discrete_dim_n=discrete_dim_n,
        traversal_n=traversal_n, total_discrete_n_to_traverse=total_discrete_n_to_traverse)
    images = get_images_from_samples(model, samples=all_latent_samples)
    pointer = 0
    plt.figure(figsize=(5, 4))
    gs = set_grid_specifications(nrows=total_discrete_n_to_traverse, ncols=traversal_n)
    for i in range(total_discrete_n_to_traverse):
        for j in range(traversal_n):
            plt.subplot(gs[i, j])
            plot_based_on_color_or_black(recon_image=images[pointer, :, :, :])
            pointer += 1
            plt.axis('off')
    plt.savefig(plots_path + f'fixed_cont_{fixed_cont_dim:d}.png')


def plot_grid_of_continuous_traversals_with_fixed_one_hot(model, fixed_cat, discrete_dim_n,
                                                          cont_dim_n, traversal_n, plots_path):
    all_latent_samples = traverse_all_cont_dim_with_fixed_discrete_dim(
        fixed_cat=fixed_cat, cont_dim_n=cont_dim_n, discrete_dim_n=discrete_dim_n,
        traversal_n=traversal_n)
    images = get_images_from_samples(model, samples=all_latent_samples)
    pointer = 0
    plt.figure(figsize=(5, 4))
    gs = set_grid_specifications(nrows=cont_dim_n, ncols=traversal_n)
    for i in range(cont_dim_n):
        for j in range(traversal_n):
            plt.subplot(gs[i, j])
            plot_based_on_color_or_black(recon_image=images[pointer, :, :, :])
            pointer += 1
            plt.axis('off')
    plt.savefig(plots_path + f'grid_for_{fixed_cat:d}.png')


def plot_grid_of_random_cont_samples_along_all_disc_dim(model, cont_dim_n, discrete_dim_n, samples_n,
                                                        plots_path, total_discrete_n_to_traverse):
    all_random_samples = sample_cont_randomly_along_all_discrete_dim(
        samples_per_cat=samples_n, discrete_dim_n=discrete_dim_n, cont_dim_n=cont_dim_n,
        total_discrete_n_to_traverse=total_discrete_n_to_traverse)
    images = get_images_from_samples(model, samples=all_random_samples)
    pointer = 0
    plt.figure(figsize=(5, 4))
    gs = set_grid_specifications(nrows=total_discrete_n_to_traverse, ncols=samples_n)
    for i in range(total_discrete_n_to_traverse):
        for j in range(samples_n):
            plt.subplot(gs[i, j])
            plot_based_on_color_or_black(recon_image=images[pointer, :, :, :])
            pointer += 1
            plt.axis('off')
    plt.savefig(plots_path + f'random_sample.png')


def set_grid_specifications(nrows, ncols):
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrows + 1), bottom=0.5 / (nrows + 1),
                           left=0.5 / (ncols + 1), right=1 - 0.5 / (ncols + 1))
    return gs


def get_images_from_samples(model, samples):
    samples = tf.concat(samples, axis=0)
    images_logits = model.decode(samples)[0]
    # images_logits = model.decode(samples)
    images = tf.math.sigmoid(images_logits)
    return images


def traverse_all_discrete_dim_with_fixed_cont_dim(fixed_cont_dim, cont_dim_n, discrete_dim_n,
                                                  traversal_n, total_discrete_n_to_traverse):
    all_latent_samples = []
    traversed_vec = get_traversed_vec(dim_to_traverse=fixed_cont_dim, dim_n=cont_dim_n,
                                      traversal_n=traversal_n)
    for disc_dim in range(total_discrete_n_to_traverse):
        one_hot = get_one_hot_vector(category_represented=disc_dim, vector_length=discrete_dim_n)
        for idx in range(traversal_n):
            latent_sample = tf.concat([traversed_vec[idx], one_hot], axis=0)
            latent_sample = tf.reshape(latent_sample, shape=(1, cont_dim_n + discrete_dim_n))
            all_latent_samples.append(latent_sample)
    return all_latent_samples


def traverse_all_cont_dim_with_fixed_discrete_dim(fixed_cat, cont_dim_n, discrete_dim_n, traversal_n):
    all_latent_samples = []
    for dim_to_traverse in range(cont_dim_n):
        one_hot = get_one_hot_vector(category_represented=fixed_cat, vector_length=discrete_dim_n)
        traversed_vec = get_traversed_vec(dim_to_traverse=dim_to_traverse, dim_n=cont_dim_n,
                                          traversal_n=traversal_n)
        for idx in range(traversal_n):
            latent_sample = tf.concat([traversed_vec[idx], one_hot], axis=0)
            latent_sample = tf.reshape(latent_sample, shape=(1, cont_dim_n + discrete_dim_n))
            all_latent_samples.append(latent_sample)
    return all_latent_samples


def get_traversed_vec(dim_to_traverse, dim_n, traversal_n):
    list_of_traversed_vectors = []
    normal_traversal_values = get_normal_distribution_traversal_values(size=traversal_n)
    for i in range(traversal_n):
        vector = np.zeros(shape=dim_n)
        vector[dim_to_traverse] = normal_traversal_values[i]
        list_of_traversed_vectors.append(vector)
    return list_of_traversed_vectors


def get_normal_distribution_traversal_values(size):
    normal_pdf_lower_bound = 0.05
    normal_pdf_upper_bound = 0.95
    cdf_traversal = np.linspace(normal_pdf_lower_bound, normal_pdf_upper_bound, size)
    normal_traversal_values = norm.ppf(cdf_traversal)
    return normal_traversal_values


def sample_cont_randomly_along_all_discrete_dim(samples_per_cat, discrete_dim_n, cont_dim_n,
                                                total_discrete_n_to_traverse):
    all_random_samples = []
    for disc_dim in range(total_discrete_n_to_traverse):
        one_hot = get_one_hot_vector(category_represented=disc_dim, vector_length=discrete_dim_n)
        cont_samples = np.random.normal(size=(samples_per_cat, cont_dim_n))
        for idx in range(samples_per_cat):
            latent_sample = tf.concat([cont_samples[idx], one_hot], axis=0)
            latent_sample = tf.reshape(latent_sample, shape=(1, cont_dim_n + discrete_dim_n))
            all_random_samples.append(latent_sample)
    return all_random_samples


def get_one_hot_vector(category_represented, vector_length):
    one_hot_vector = np.zeros(shape=vector_length)
    one_hot_vector[category_represented] = 1.
    return one_hot_vector


def plot_grid_of_generated_digits(model, n_required: int, fig_size=10, digit_size=28, std_dev=2.,
                                  filename='./Results/vae.html'):
    figure = np.zeros((digit_size * fig_size, digit_size * fig_size))
    grid_x = np.linspace(-std_dev, std_dev, fig_size)
    grid_y = np.linspace(-std_dev, std_dev, fig_size)

    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y):
            if model.disc_latent_n > 0:
                z_sample = np.random.normal(loc=0, scale=1, size=model.cont_latent_n)
                c_sample = np.zeros(model.disc_latent_n * model.disc_var_num)
                c_sample[j % n_required] = 1.
                latent_sample = np.hstack((z_sample, c_sample))
            else:
                latent_sample = np.zeros(shape=model.cont_latent_n)
                latent_sample[0], latent_sample[1] = xi, yi

            digit = generate_digit_from_latent_sample(model, latent_sample, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

    trace = go.Heatmap(x=grid_x, y=grid_y, z=figure, colorscale='Viridis')
    layout = go.Layout(yaxis=dict(autorange='reversed'))
    fig = go.Figure(data=[trace], layout=layout)
    py.plot(fig, filename=filename, auto_open=False)


def generate_digit_from_latent_sample(model, latent_sample, digit_size):
    latent_sample = np.broadcast_to(latent_sample, shape=(1, latent_sample.shape[0]))
    generated = model.decode(latent_sample)[0].numpy()
    # generated = model.decode(latent_sample).numpy()
    digit = generated[0, :, :, 0].reshape(digit_size, digit_size)
    digit = np.exp(digit) / (1 + np.exp(digit))
    # digit[digit >= 0.5] = 1
    # digit[digit < 0.5] = 0
    return digit

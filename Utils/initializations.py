# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Imports
# ===========================================================================================================
import numpy as np
import tensorflow as tf
from typing import Tuple
from os import environ as os_env
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Initialization functions
# ===========================================================================================================


def get_uniform_mix_probs(initial_point: int, middle_point: int, final_point: int, mass_in_beginning,
                          max_size: int) -> np.ndarray:
    probs = np.zeros(shape=max_size)
    points_in_beginning_n = (middle_point - initial_point + 1)
    points_in_end_n = (final_point - middle_point)
    mass_in_end = 1. - mass_in_beginning

    for i in range(final_point - initial_point + 1):
        if i <= middle_point:
            probs[i] = mass_in_beginning / points_in_beginning_n
        else:
            probs[i] = mass_in_end / points_in_end_n
    return probs


def sample_from_uniform_mix(size: int, initial_point: int, middle_point: int, final_point: int,
                            mass_in_beginning):
    first_samples = np.random.randint(low=initial_point, high=middle_point, size=size)
    mixture_samples = np.random.randint(low=middle_point, high=final_point + 1, size=size)
    prob_of_beginning = np.random.uniform(low=0, high=1, size=size)
    samples_in_beginning = np.where(prob_of_beginning > 1. - mass_in_beginning)[0]
    mixture_samples[samples_in_beginning] = first_samples[samples_in_beginning]
    return mixture_samples


def initialize_mu_and_xi_for_logistic(shape, seed: int = 21) -> Tuple[tf.Variable, tf.Variable]:
    categories = shape[1]
    np.random.RandomState(seed=seed)
    inv_softplus_one = 0.541324854612918
    mu = np.random.normal(loc=0, scale=0.01, size=shape)
    xi = np.random.normal(loc=inv_softplus_one, scale=0.01, size=shape)
    a = 1 / categories
    if a == 1:
        mu[:, :, :] = 0.
    else:
        mu[:, :categories, :] = tf.math.log(a / (1 - a))

    mu = tf.Variable(initial_value=mu, dtype=tf.float32)
    xi = tf.Variable(initial_value=xi, dtype=tf.float32)
    return mu, xi


def get_initial_analytical_mus(probs: np.ndarray, initialization_type: str = 'stick-break') -> tf.Tensor:
    mu, cumsum = np.zeros(shape=probs.shape[0]), 0

    if initialization_type == 'stick-break':
        for i in range(probs.shape[0]):
            mu[i] = np.log(probs[i] / (1. - cumsum))
            cumsum += probs[i]

    elif initialization_type == 'sigmoid':
        for i in range(probs.shape[0]):
            mu[i] = np.log(probs[i] / (1. - probs[i]))

    return tf.reshape(tf.constant(value=mu, dtype=tf.float32), shape=(probs.shape[0], 1))
# ===========================================================================================================

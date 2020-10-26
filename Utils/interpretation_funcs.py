import numpy as np
from scipy.stats import norm as gaussian
from scipy.integrate import quad


def compute_softmax_pp_prob(mu, sigma):
    categories_n = mu.shape[0]
    cat_probs = np.zeros(categories_n + 1)
    for k in range(categories_n):
        cat_probs[k] = compute_individual_cat_probs(k, mu, sigma, use_pp=True)
    cat_probs[categories_n] = 1 - np.sum(cat_probs)
    return cat_probs


def compute_cat_prob(mu, sigma, from_sample, sample_size=int(1.e2)):
    categories_n = mu.shape[0]
    cat_probs = np.zeros(categories_n)
    for k in range(categories_n - 1):
        cat_probs[k] = compute_individual_cat_probs(k, mu, sigma,
                                                    from_sample, sample_size,
                                                    use_pp=False)
    cat_probs[categories_n - 1] = 1 - np.sum(cat_probs)
    return cat_probs


def compute_individual_cat_probs(k, mu, sigma, from_sample=False, sample_size=int(1.e2), use_pp=True):
    if from_sample:
        cat_prob = compute_cat_expectation(k, mu, sigma, sample_size)
    else:
        cat_prob = compute_cat_integral(k, mu, sigma, use_pp)
    return cat_prob


def compute_cat_expectation(k, mu, sigma, sample_size):
    cat_prob = 1.
    z = np.random.normal(mu[k], sigma[k], size=sample_size)
    cat_prob = compute_more_than(z, k, mu, sigma, use_samples=True)
    return np.mean(cat_prob)


def compute_cat_integral(k, mu, sigma, use_pp):
    cat_prob = 1.
    lower = 0 if use_pp else -np.inf
    upper = np.inf
    cat_prob, _ = quad(compute_more_than, lower, upper, args=(k, mu, sigma, False))
    cat_prob
    return cat_prob


def compute_more_than(z, k, mu, sigma, use_samples):
    more_than = 1.
    categories_n = mu.shape[0]
    for j in range(categories_n):
        if j != k:
            more_than *= gaussian.cdf(z, loc=mu[j], scale=sigma[j])
    if not use_samples:
        more_than *= gaussian.pdf(z, loc=mu[k], scale=sigma[k])
        # more_than *= gaussian.cdf(-z, loc=mu[k], scale=sigma[k])
    return more_than

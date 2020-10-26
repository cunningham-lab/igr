import numpy as np
from Utils.interpretation_funcs import compute_cat_prob
from Utils.interpretation_funcs import compute_softmax_pp_prob


def test_softmax_pp_inter():
    tolerance = 1.e-1
    samples_n = int(1.e6)
    mu = np.array([3., 2., 1.])
    sigma = np.array([4., 5., 1.])
    cov = np.diag(sigma ** 2)
    categories_n = mu.shape[0]
    z = np.random.multivariate_normal(mu, cov, size=samples_n)

    w_argmax = np.argmax(z, axis=1)
    ans = np.zeros(categories_n + 1)
    for k in range(categories_n):
        ind_k = w_argmax == k
        total_positive = ind_k.shape[0]
        count_positive = np.sum(z[ind_k, k] > 0)
        ans[k] = count_positive / total_positive
    ans[categories_n] = 1 - np.sum(ans[:categories_n])

    approx = compute_softmax_pp_prob(mu, sigma)

    diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
    print(f'\nDiff {diff:1.2e}')
    assert diff < tolerance


def test_gaussian_inter():
    tolerance = 1.e-1
    mu = np.array([3., 2., 1., 0.1])
    sigma = np.array([0.1, 1., 1., 3])
    cov = np.diag(sigma ** 2)
    samples_n = int(1.e4)
    categories_n = mu.shape[0]
    z = np.random.multivariate_normal(mu, cov, size=samples_n)
    from_sample = True
    inner_sample_size = int(1.e2)

    w_argmax = np.argmax(z, axis=1)
    ans = np.zeros(categories_n)
    for k in range(categories_n):
        ans[k] = np.mean(w_argmax == k)
    approx = compute_cat_prob(mu, sigma, from_sample, inner_sample_size)

    diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
    print(f'\nDiff {diff:1.2e}')
    assert diff < tolerance


def test_gaussian_inter_uniform():
    categories_n = 9
    samples_n = int(1.e4)
    tolerance = 1.e-1

    mu = np.zeros(categories_n)
    # mu = np.zeros(categories_n) - 0.754
    sigma = np.ones(categories_n)
    z = np.random.multivariate_normal(mu, np.diag(sigma), size=samples_n)

    w_argmax = np.argmax(z, axis=1)
    ans = np.zeros(categories_n)
    for k in range(categories_n):
        ans[k] = np.mean(w_argmax == k)
    approx = compute_cat_prob(mu, sigma, from_sample=False)

    diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
    print(f'\nDiff {diff:1.2e}')
    assert diff < tolerance

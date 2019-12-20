# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Imports
# ===========================================================================================================
import unittest
import numpy as np
from Utils.mmd import compute_mmd2biased, compute_mmd2u, generate_null_distribution_by_bootstrap, apply_kernel
from Utils.mmd import compute_l2_distances, calculate_median_bandwidth
from sklearn.metrics import pairwise_kernels
from multiprocessing import Pool
# ===========================================================================================================


class TestSBDist(unittest.TestCase):

    def test_compute_mmd2biased(self):
        #    Global parameters  #######
        test_tolerance = 1.e-10
        bandwidth = np.sqrt(0.5)
        np.random.seed(seed=21)
        samples_n = int(1.e2)

        p_samples = np.random.laplace(loc=0, scale=1, size=samples_n)
        q_samples = np.random.laplace(loc=0, scale=1, size=samples_n)
        kernel = np.zeros(shape=(samples_n, samples_n))
        for i in range(samples_n):
            for j in range(samples_n):
                kernel[i, j] = (apply_kernel(v=p_samples[i] - p_samples[j], bandwidth=bandwidth) +
                                apply_kernel(v=q_samples[i] - q_samples[j], bandwidth=bandwidth) -
                                2 * apply_kernel(v=q_samples[i] - p_samples[j], bandwidth=bandwidth))
        mmd_ans = np.mean(kernel)

        mmd = compute_mmd2biased(p_samples=p_samples, q_samples=q_samples, bandwidth=bandwidth)
        relative_diff = np.abs(mmd - mmd_ans) / np.abs(mmd_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_compute_mmd2u(self):
        #    Global parameters  #######
        test_tolerance = 1.e-10
        p = np.array([-0.1, -0.2, -0.3])
        q = np.array([0.1, 0.2, 0.3])
        bandwidth = np.sqrt(0.5)

        mmd_unbiased_ans = ((1 / (3 * 2)) * (ker(-0.1, -0.2) + ker(-0.1, -0.3) +
                                             ker(-0.2, -0.1) + ker(-0.2, -0.3) +
                                             ker(-0.3, -0.1) + ker(-0.3, -0.2)) +
                            (1 / (3 * 2)) * (ker(0.1, 0.2) + ker(0.1, 0.3) +
                                             ker(0.2, 0.1) + ker(0.2, 0.3) +
                                             ker(0.3, 0.1) + ker(0.3, 0.2)) +
                            (-2 / 9) * (ker(-0.1, 0.1) + ker(-0.1, 0.2) + ker(-0.1, 0.3) +
                                        ker(-0.2, 0.1) + ker(-0.2, 0.2) + ker(-0.2, 0.3) +
                                        ker(-0.3, 0.1) + ker(-0.3, 0.2) + ker(-0.3, 0.3)))

        mmd2u_pack, *_ = kernel_two_sample_test(x=p.reshape(-1, 1), y=q.reshape(-1, 1))
        mmd2u = compute_mmd2u(p_samples=p, q_samples=q, bandwidth=bandwidth)

        relative_diff = np.abs(mmd2u - mmd_unbiased_ans) / np.abs(mmd_unbiased_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)
        relative_diff = np.abs(mmd2u - mmd2u_pack) / np.abs(mmd2u_pack)
        self.assertTrue(expr=relative_diff < test_tolerance)

    def test_compare_mmd2u_and_p_val_vs_package(self):
        #    Global parameters  #######
        test_tolerance = 1.e-7
        small_mmd = 1.e-2
        large_mmd = 0.1
        pool = Pool()
        np.random.seed(seed=21)
        bandwidth = np.sqrt(0.5)  # so that the kernel width is 1. - this is just standardization
        histogram_samples = 1_000

        # Testing for the same distribution with provided implementation
        samples_n = int(1.e2)
        p_samples = np.random.laplace(loc=0, scale=1, size=samples_n)
        q_samples = np.random.laplace(loc=0, scale=1, size=samples_n)

        mmd2u = compute_mmd2u(p_samples=p_samples, q_samples=q_samples, bandwidth=bandwidth)
        null_dist = generate_null_distribution_by_bootstrap(total_samples=histogram_samples,
                                                            p_samples=p_samples, q_samples=q_samples,
                                                            bandwidth=bandwidth, pool=pool)
        p_val = np.mean(null_dist >= mmd2u)
        mmd2u_package, _, p_val_package = kernel_two_sample_test(x=p_samples.reshape(-1, 1),
                                                                 y=q_samples.reshape(-1, 1))

        #    Compute differences  #######
        relative_diff = np.abs(mmd2u - mmd2u_package) / np.abs(mmd2u_package)
        self.assertTrue(expr=relative_diff < test_tolerance)

        relative_diff = np.abs(p_val - p_val_package)
        self.assertTrue(expr=relative_diff < 0.1)

        self.assertTrue(expr=mmd2u < small_mmd)

        # Testing for a different distribution with provided implementation
        p_samples = np.random.laplace(loc=0, scale=1, size=samples_n)
        q_samples = np.random.exponential(scale=5., size=samples_n)
        mmd2u = compute_mmd2u(p_samples=p_samples, q_samples=q_samples, bandwidth=bandwidth)
        null_dist = generate_null_distribution_by_bootstrap(total_samples=histogram_samples,
                                                            p_samples=p_samples, q_samples=q_samples,
                                                            bandwidth=bandwidth, pool=pool)
        p_val = np.mean(null_dist >= mmd2u)
        mmd2u_package, _, p_val_package = kernel_two_sample_test(x=p_samples.reshape(-1, 1),
                                                                 y=q_samples.reshape(-1, 1))

        #    Compute differences  #######
        relative_diff = np.abs(mmd2u - mmd2u_package) / np.abs(mmd2u_package)
        self.assertTrue(expr=relative_diff < test_tolerance)

        relative_diff = np.abs(p_val - p_val_package)
        self.assertTrue(expr=relative_diff < 0.1)

        self.assertTrue(expr=mmd2u > large_mmd)

    def test_calculate_median_bandwidth(self):
        #    Global parameters  #######
        test_tolerance = 1.e-8

        #   Test    ###################
        p_samples = np.array([1, 2, 3])
        q_samples = np.array([-1, 7, 6, -2])

        dist = compute_l2_distances(p_samples, q_samples)
        dist_ans = np.array([[4., 36., 25., 9.],
                             [9., 25., 16., 16.],
                             [16., 16., 9., 25.]])

        #    Compute differences  #######
        relative_diff = np.linalg.norm(dist_ans - dist) / np.linalg.norm(dist_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)

        median_dist = calculate_median_bandwidth(p_samples=p_samples, q_samples=q_samples)
        median_dist_ans = np.sqrt(np.median(dist_ans) / 2)

        #    Compute differences  #######
        relative_diff = np.linalg.norm(median_dist - median_dist_ans) / np.linalg.norm(median_dist_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)


def ker(v, w, bandwidth=np.sqrt(0.5)):
    return np.exp(-0.5 * ((v - w)/bandwidth) ** 2)


def kernel_two_sample_test(x, y, kernel_function="rbf", iterations=10000, verbose=False, random_state=None,
                           **kwargs):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.
    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function=‘rbf’, gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(x)
    n = len(y)
    xy = np.vstack([x, y])
    k = pairwise_kernels(xy, metric=kernel_function, **kwargs)
    mmd2u = calculate_mmd2u(k, m, n)
    if verbose:
        print('MMD^2_u = %s' % mmd2u)
        print('Computing the null distribution.')

    mmd2u_null = compute_null_distribution(k, m, n, iterations, verbose=verbose, random_state=random_state)
    p_value = max(1.0 / iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))
    if verbose:
        print('p-value ~= %s \t (resolution : %s)' % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value


def compute_null_distribution(k, m, n, iterations=10000, verbose=False, random_state=None,
                              marker_interval=1000):
    """""Compute the bootstrap null-distribution of calculate_mmd2u.
    """""
    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i)
        idx = rng.permutation(m + n)
        k_i = k[idx, idx[:, None]]
        mmd2u_null[i] = calculate_mmd2u(k_i, m, n)

    if verbose:
        print('')

    return mmd2u_null


def calculate_mmd2u(k, m, n):
    """The MMD^2_u unbiased statistic.
    """
    kx = k[:m, :m]
    ky = k[m:, m:]
    kxy = k[:m, m:]
    output = (1.0 / (m * (m - 1.0)) * (kx.sum() - kx.diagonal().sum()) +
              1.0 / (n * (n - 1.0)) * (ky.sum() - ky.diagonal().sum()) -
              2.0 / (m * n) * kxy.sum())
    return output


if __name__ == '__main__':
    unittest.main()

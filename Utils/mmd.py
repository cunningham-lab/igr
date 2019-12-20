# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Imports
# ===========================================================================================================
import time
import numpy as np
import numba
from functools import partial
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# MMD functions below
# ===========================================================================================================


def generate_null_distribution_by_bootstrap(total_samples, p_samples, q_samples, bandwidth, pool):
    t0 = time.time()
    num_p_samples = p_samples.shape[0]
    bootstrap_permutations = np.random.choice(a=2 * num_p_samples, size=(total_samples, 2 * num_p_samples))

    func = partial(compute_bootstrap_mmd, p_samples=p_samples, q_samples=q_samples,
                   bootstrap_permutations=bootstrap_permutations, bandwidth=bandwidth)
    mmd_multiprocess = pool.map(func=func, iterable=[sample for sample in range(total_samples)])
    null_dist = np.array(mmd_multiprocess)

    print(f'\nGenerating Null Distribution took: {time.time() - t0:6.1f} sec')
    return null_dist


def compute_bootstrap_mmd(sample: int, p_samples, q_samples, bootstrap_permutations, bandwidth):
    total_samples = p_samples.shape[0]
    samples_stack = np.hstack((p_samples, q_samples))
    p_samples_boot = samples_stack[bootstrap_permutations[sample, :total_samples]]
    q_samples_boot = samples_stack[bootstrap_permutations[sample, total_samples:]]

    mmd = compute_mmd2u(p_samples=p_samples_boot, q_samples=q_samples_boot, bandwidth=bandwidth)
    return mmd


@numba.jit(nopython=True)
def compute_mmd2u(p_samples: np.ndarray, q_samples: np.ndarray, bandwidth: float) -> np.ndarray:
    num_p_samples = p_samples.shape[0]
    num_q_samples = q_samples.shape[0]
    normalize_kpp = 1 / (num_p_samples * (num_p_samples - 1))
    normalize_kqq = 1 / (num_q_samples * (num_q_samples - 1))
    normalize_kpq = 2 / (num_p_samples * num_q_samples)
    kernel_pp = np.zeros(shape=(num_p_samples, num_p_samples))
    kernel_qq = np.zeros(shape=(num_q_samples, num_q_samples))
    kernel_pq = np.zeros(shape=(num_p_samples, num_q_samples))

    for i in range(num_p_samples):
        for j in range(num_q_samples):
            if i == j:
                kernel_pp[i, j] = 0
                kernel_qq[i, j] = 0
                kernel_pq[i, j] = np.exp(-0.5 * (p_samples[i] - q_samples[j]) ** 2 / bandwidth ** 2)
            else:
                kernel_pp[i, j] = np.exp(-0.5 * (p_samples[i] - p_samples[j]) ** 2 / bandwidth ** 2)
                kernel_qq[i, j] = np.exp(-0.5 * (q_samples[i] - q_samples[j]) ** 2 / bandwidth ** 2)
                kernel_pq[i, j] = np.exp(-0.5 * (p_samples[i] - q_samples[j]) ** 2 / bandwidth ** 2)

    mmd = (normalize_kpp * np.sum(kernel_pp) +
           normalize_kqq * np.sum(kernel_qq) -
           normalize_kpq * np.sum(kernel_pq))

    return mmd


def compute_mmd2biased(p_samples: np.ndarray, q_samples: np.ndarray, bandwidth: float) -> np.ndarray:

    kernel_pp = apply_kernel(broadcast_diff_between(p_samples, p_samples), bandwidth=bandwidth)
    kernel_qq = apply_kernel(broadcast_diff_between(q_samples, q_samples), bandwidth=bandwidth)
    kernel_pq = apply_kernel(broadcast_diff_between(p_samples, q_samples), bandwidth=bandwidth)

    mmd = np.mean(kernel_pp + kernel_qq - 2 * kernel_pq)
    return mmd


def broadcast_diff_between(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    v_broad = np.broadcast_to(array=v, shape=(v.shape[0], w.shape[0]))
    diff = w.reshape(w.shape[0], 1) - v_broad
    return diff


def apply_kernel(v: np.ndarray, bandwidth: float):
    output = np.exp(-0.5 * (v / bandwidth) ** 2)
    return output


def calculate_median_bandwidth(p_samples: np.ndarray, q_samples: np.ndarray):
    l2_distances = compute_l2_distances(p_samples, q_samples)
    median_bandwidth = np.sqrt(np.median(l2_distances) / 2)
    return median_bandwidth


@numba.jit(nopython=True, parallel=True)
def compute_l2_distances(v, w):
    n = v.shape[0]
    m = w.shape[0]
    l2_distances = np.zeros(shape=(n, m))
    for i in numba.prange(n):
        for j in range(m):
            l2_distances[i, j] = (v[i] - w[j]) ** 2
    return l2_distances

# ===========================================================================================================

import numpy as np
from matplotlib import pyplot as plt
from Utils.MinimizeEmpiricalLoss import MinimizeEmpiricalLoss, get_initial_params_for_model_type
from Utils.general import get_for_approx
from Utils.general import sample_from_approx
# from Utils.general import plot_initial_final_histograms

sample_size = int(1.e4)
total_iterations = 2 * int(1.e2)
learning_rate = 1.e-2

samples_plot_n = int(1.e3)
batch_n = 1
np.random.RandomState(seed=21)
temp = 0.01
threshold = 0.9
# categories_n = 200
categories_n = 13
shape = (batch_n, categories_n, 1, 1)
model_type = 'IGR_SB_Finite'
run_against = 'binomial'
x_lim_max, y_lim_max = 70, 0.2
plot_with_loss = False

probs, p_samples, results_file = get_for_approx(run_against, categories_n, samples_plot_n)

# Train model
# ===============================================================================================
params, params_init = get_initial_params_for_model_type(model_type=model_type, shape=shape)
minimizer = MinimizeEmpiricalLoss(learning_rate=learning_rate, temp=temp, sample_size=sample_size,
                                  run_kl=False, params=params, max_iterations=total_iterations,
                                  model_type=model_type, threshold=threshold,
                                  planar_flow=None)
minimizer.run_iteratively = True
minimizer.optimize_model(probs=probs)

# Run samples
# ================================================================================================
q_samples, q_samples_init = sample_from_approx(params, params_init, temp, model_type, p_samples,
                                               samples_plot_n, minimizer.threshold,
                                               minimizer.planar_flow)

# Plot Loss and Histograms
# ================================================================================================
# plt.style.use(style='ggplot')
#
# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=150)
# plot_initial_final_histograms(ax=ax, p_samples=p_samples,
#                               q_samples=q_samples, model_type=model_type,
#                               y_lim_max=y_lim_max, x_lim_max=x_lim_max,
#                               q_samples_init=q_samples_init,
#                               number_of_bins=x_lim_max + 2)
#
# plt.tight_layout()
# plt.savefig(fname='./results/plot.pdf', bbox_inches='tight')
# plt.show()

plt.figure(figsize=(5, 4))
plt.hist(p_samples, bins=np.arange(11), density=True, alpha=0.5, color='gray', label='p')
plt.hist(q_samples, bins=np.arange(11), density=True, alpha=0.5, color='blue', label='IGR-SB')
# plt.hist(q_samples_init, bins=np.arange(11), density=True, alpha=0.5, color='blue', label='IGR-SB')
plt.legend()
plt.ylim([0, 0.3])
# plt.title('Initial distribution')
plt.title('Final distribution')
plt.tight_layout()
plt.savefig('igr_hist_binomial.pdf')
# plt.savefig('igr_hist_discrete.pdf')
# plt.savefig('igr_hist_discrete_init.pdf')
plt.show()

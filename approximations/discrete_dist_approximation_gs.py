from matplotlib import pyplot as plt
from Utils.MinimizeEmpiricalLoss import MinimizeEmpiricalLoss, get_initial_params_for_model_type
from Utils.general import plot_histograms_of_gs, get_for_approx, sample_from_approx, offload_case

sample_size = int(1.e4)
total_iterations = 2 * int(1.e2)
learning_rate = 1.e-2
samples_plot_n = int(1.e4)
batch_n = 1
temp = 0.1
model_type = 'GS'

cases = {
    1: {'run_against': 'discrete', 'categories_n_list': [10], 'y_lim_max': 0.3, 'x_lim_max': 10,
        'categories_n': 10},
    2: {'run_against': 'binomial', 'categories_n_list': [12], 'y_lim_max': 0.3, 'x_lim_max': 12,
        'categories_n': 12},
    3: {'run_against': 'poisson', 'categories_n_list': [20, 40, 100],
        'y_lim_max': 0.2, 'x_lim_max': 70, 'categories_n': 100},
    4: {'run_against': 'negative_binomial', 'categories_n_list': [20, 40, 100],
        'y_lim_max': 0.2, 'x_lim_max': 70, 'categories_n': 100}
}
selected_case = 3
run_against, categories_n, categories_n_list, y_lim_max, x_lim_max = offload_case(
    cases[selected_case])
probs, p_samples, _ = get_for_approx(run_against, categories_n, samples_plot_n)

# Train model
# =====================================================================================================
q_samples_list = []
q_samples_init_list = []
for i in range(len(categories_n_list)):
    categories_n = categories_n_list[i]
    shape = (batch_n, categories_n, 1, 1)
    params, params_init = get_initial_params_for_model_type(model_type=model_type, shape=shape)

    minimizer = MinimizeEmpiricalLoss(learning_rate=learning_rate, temp=temp, sample_size=sample_size,
                                      run_kl=True, params=params,
                                      max_iterations=total_iterations, model_type=model_type)
    minimizer.optimize_model(probs=probs)
    q_samples, q_samples_init = sample_from_approx(params, params_init, temp, model_type, p_samples,
                                                   samples_plot_n, minimizer.threshold)
    q_samples_list.append(q_samples)
    q_samples_init_list.append(q_samples_init)

# Plot Loss and Histograms
# =====================================================================================================
plt.style.use(style='ggplot')
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=150)
plot_histograms_of_gs(ax=ax, p_samples=p_samples, q_samples_list=q_samples_list,
                      q_samples_init_list=q_samples_init_list, number_of_bins=categories_n + 2,
                      categories_n_list=categories_n_list, y_lim_max=y_lim_max, x_lim_max=x_lim_max)
plt.tight_layout()
plt.savefig(fname='./Results/gs_hist.pdf', bbox_inches='tight')
plt.show()

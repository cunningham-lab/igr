import pickle
import numpy as np
from matplotlib import pyplot as plt
from Utils.MinimizeEmpiricalLoss import MinimizeEmpiricalLoss, get_initial_params_for_model_type
from Utils.general import get_for_approx, plot_loss_and_initial_final_histograms, sample_from_approx

sample_size = 1000
total_iterations = 1 * int(1.e2)
# total_iterations = 1 * int(1.e2)
learning_rate = 1.e-2

samples_plot_n = int(1.e4)
batch_n = 1
np.random.RandomState(seed=21)
temp = 0.01
threshold = 0.9
# categories_n = 10
categories_n = 200
# categories_n = 12
shape = (batch_n, categories_n, 1, 1)
# model_type = 'IGR_I'
model_type = 'IGR_SB'
# model_type = 'IGR_SB_Finite'
skip_first_iterations = 10

# run_against = 'uniform'
run_against = 'poisson'
x_lim_max, y_lim_max = 70, 0.2
save_parameters = False

probs, p_samples, results_file = get_for_approx(run_against, categories_n, samples_plot_n)

# Train model
# ===========================================================================================================
params, params_init = get_initial_params_for_model_type(model_type=model_type, shape=shape)

minimizer = MinimizeEmpiricalLoss(learning_rate=learning_rate, temp=temp, sample_size=sample_size,
                                  run_kl=False, params=params,
                                  max_iterations=total_iterations, model_type=model_type, threshold=threshold)
minimizer.run_iteratively = True
minimizer.optimize_model(probs=probs)

if save_parameters:
    with open(file=results_file, mode='wb') as f:
        if model_type in ['IGR_I', 'IGR_SB', 'IGR_SB_Finite']:
            pickle.dump(obj={'mu': params[0].numpy(), 'xi': params[1].numpy()}, file=f)

# Run samples
# ===========================================================================================================
q_samples, q_samples_init = sample_from_approx(params, params_init, temp, model_type, p_samples,
                                               samples_plot_n, minimizer.threshold)

# Plot Loss and Histograms
# ===========================================================================================================
plt.style.use(style='ggplot')
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))

plot_loss_and_initial_final_histograms(
    ax=ax, p_samples=p_samples, q_samples=q_samples,
    loss_iter=minimizer.loss_iter[skip_first_iterations:minimizer.iteration],
    title=f'Empirical KL Loss for {model_type}', model_type=model_type,
    y_lim_max=y_lim_max, x_lim_max=x_lim_max,
    q_samples_init=q_samples_init,
    number_of_bins=x_lim_max + 2)

plt.tight_layout()
plt.savefig(fname='./Results/plot.png')
plt.show()

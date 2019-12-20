import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from Models.VAENet import setup_model
from Models.train_vae import setup_vae_optimizer
from GS_vs_SB_analysis.simplex_proximity_funcs import calculate_distance_to_simplex
from Utils.load_data import load_vae_dataset

dataset_name = 'fmnist'
run_with_sample = False
model_type = 'PlanarDis'
plots_path = './Results/Outputs/'
path_to_results = './Results/Current_Model/'
# hyper_file = 'hyper_planar2.pkl'
# weights_file = 'vae_planar2.h5'
# hyper_file = 'hyper_planar5.pkl'
# weights_file = 'vae_planar5.h5'
hyper_file = 'hyper_planar10.pkl'
weights_file = 'vae_planar10.h5'
# hyper_file = 'hyper_test.pkl'
# weights_file = 'vae_test.h5'
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
    mu, xi = params
    batch_size, categories_n, _, num_of_vars = mu.shape
    shape_broad = (batch_size, categories_n, samples_n, num_of_vars)
    params_broad = []
    for param in params:
        param_w_samples = tf.broadcast_to(input=param, shape=shape_broad)
        params_broad.append(param_w_samples)
    mu_broad, xi_broad = params_broad
    epsilon = tf.random.normal(shape=mu_broad.shape)
    sigma_broad = tf.math.exp(xi_broad)
    lam = vae_opt.model.planar_flow(mu_broad + sigma_broad * epsilon)
    ψ = tf.math.softmax(lam / vae_opt.temp, axis=1).numpy()
    for i in range(ψ.shape[0]):
        for k in range(ψ.shape[3]):
            diff[im_idx, :, k] = calculate_distance_to_simplex(
                ψ=ψ[i, :, :, k], argmax_locs=np.argmax(ψ[i, :, :, k], axis=0))
        im_idx += 1
diff_planar = diff.copy()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Make boxplot
# ===========================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=100)

rows_list = []
model_name = ['Planar10']
diff_list = [np.median(np.mean(diff_planar, axis=2), axis=1)]
for i in range(len(diff_list)):
    for s in range(total_test_images):
        entry = {'Model': model_name[i], 'Distance': diff_list[i][s]}
        rows_list.append(entry)
df = pd.DataFrame(rows_list)
ax = sns.boxplot(x='Model', y='Distance', data=df, color='royalblue', boxprops={'alpha': 0.5})

plt.ylabel('Euclidean Distance')
plt.xlabel('Models')
plt.legend()
plt.savefig('./Results/Outputs/posterior_samples.png')
plt.tight_layout()
plt.show()
# ===========================================================================================================

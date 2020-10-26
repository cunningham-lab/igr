import time
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
from Models.train_vae import construct_nets_and_optimizer
from Utils.load_data import load_vae_dataset

tic = time.time()
dataset_name = 'mnist'
path_to_trained_models = './Results/trained_models/' + dataset_name + '/'
models = {
    1: {'model_dir': 'relax_igr', 'model_type': 'Relax_IGR'},
    2: {'model_dir': 'relax_gs', 'model_type': 'Relax_GS_Dis'},
}
select_case = 1
hyper_file = 'hyper.pkl'
weights_file = 'vae.h5'
model_type = models[select_case]['model_type']
path_to_trained_models += models[select_case]['model_dir'] + '/'

with open(file=path_to_trained_models + hyper_file, mode='rb') as f:
    hyper = pickle.load(f)

batch_n = hyper['batch_n']
tf.random.set_seed(seed=hyper['seed'])
data = load_vae_dataset(dataset_name=dataset_name, batch_n=batch_n, epochs=hyper['epochs'],
                        run_with_sample=False,
                        architecture=hyper['architecture'], hyper=hyper)
_, _, np_test_images, hyper = data
vae_opt = construct_nets_and_optimizer(hyper=hyper, model_type=model_type)
vae_opt.nets.load_weights(filepath=path_to_trained_models + weights_file)

x = tf.constant(np_test_images, dtype=tf.float32)
params = vae_opt.nets.encode(x)
vae_opt.offload_params(params)
one_hot = vae_opt.get_relax_variables_from_params(x, params)[-1]
x_logit = vae_opt.decode([one_hot])
recon_probs = tf.math.sigmoid(x_logit)
plt.figure(figsize=(5, 4), dpi=100)
for i in range(np_test_images.shape[0]):
    plt.subplot(5, 4, i + 1)
    plt.imshow(recon_probs[i, :, :, 0, 0], cmap='gray')
    plt.axis('off')
plt.show()

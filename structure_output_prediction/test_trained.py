import time
import pickle
import tensorflow as tf
from Models.SOPOptimizer import setup_sop_optimizer
from Models.SOPOptimizer import evaluate_loss_on_batch
from Utils.load_data import load_mnist_sop_data

tic = time.time()
dataset_name = 'sop'
path_to_trained_models = './Results/trained_models/' + dataset_name + '/'
models = {
    1: {'model_dir': 'igr', 'model_type': 'IGR_I'},
    2: {'model_dir': 'gs', 'model_type': 'GS'},
    3: {'model_dir': 'pf', 'model_type': 'IGR_Planar'},
    4: {'model_dir': 'sb', 'model_type': 'IGR_SB_Finite'},
}
select_case = 2
run_with_sample = False
# samples_n = 1 * int(1.e3)
samples_n = 1 * int(1.e3)

hyper_file, weights_file = 'hyper.pkl', 'w.h5'
model_type = models[select_case]['model_type']
path_to_trained_models += models[select_case]['model_dir'] + '/'

with open(file=path_to_trained_models + hyper_file, mode='rb') as f:
    hyper = pickle.load(f)

batch_n = hyper['batch_size']
hyper['test_sample_size'] = samples_n
tf.random.set_seed(seed=hyper['seed'])
data = load_mnist_sop_data(batch_n=hyper['batch_size'], run_with_sample=run_with_sample)
train_dataset, test_dataset = data
epoch = hyper['epochs']
sop_optimizer = setup_sop_optimizer(hyper=hyper)
for x in train_dataset:
    x_upper = x[:, :14, :, :]
    break
sop_optimizer.batch_n = batch_n
aux = sop_optimizer.model.call(x_upper)
sop_optimizer.model.load_weights(filepath=path_to_trained_models + weights_file)

test_loss_mean = tf.keras.metrics.Mean()
for x in test_dataset:
    loss = evaluate_loss_on_batch(x, sop_optimizer, hyper)
    test_loss_mean(loss)

evaluation_print = f'Epoch {epoch:4d} || '
evaluation_print += f'TeNLL {-test_loss_mean.result():2.5e} || '
# evaluation_print += f'Train Loss {train_loss_mean.result():2.5e} || '
evaluation_print += f'{model_type} || '
toc = time.time()
evaluation_print += f'Time: {toc - tic:2.2e} sec'
print(evaluation_print)

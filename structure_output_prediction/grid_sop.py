from Utils.load_data import load_mnist_sop_data
from Models.SOPOptimizer import run_sop
import tensorflow as tf

model_type = 'IGR'
hyper = {'width_height': (14, 28, 1),
         'model_type': model_type,
         'batch_size': 64, 'learning_rate': 0.0003,
         'epochs': 100, 'iter_per_epoch': 937, 'temp': tf.constant(0.5)}
data = load_mnist_sop_data(batch_n=hyper['batch_size'], epochs=hyper['epochs'])

experiment = {
    1: {'model_type': 'IGR', 'temp': tf.constant(0.5), 'learning_rate': 0.0003},
    2: {'model_type': 'GS', 'temp': tf.constant(0.67), 'learning_rate': 0.0003},

    3: {'model_type': 'IGR', 'temp': tf.constant(0.15), 'learning_rate': 0.0003},
    4: {'model_type': 'GS', 'temp': tf.constant(0.10), 'learning_rate': 0.0003},

    5: {'model_type': 'IGR', 'temp': tf.constant(0.5), 'learning_rate': 0.001},
    6: {'model_type': 'GS', 'temp': tf.constant(0.67), 'learning_rate': 0.001},

    7: {'model_type': 'IGR', 'temp': tf.constant(0.15), 'learning_rate': 0.001},
    8: {'model_type': 'GS', 'temp': tf.constant(0.10), 'learning_rate': 0.001},

    9: {'model_type': 'IGR', 'temp': tf.constant(0.5), 'learning_rate': 0.0001},
    10: {'model_type': 'GS', 'temp': tf.constant(0.67), 'learning_rate': 0.0001},

    11: {'model_type': 'IGR', 'temp': tf.constant(0.15), 'learning_rate': 0.0001},
    12: {'model_type': 'GS', 'temp': tf.constant(0.10), 'learning_rate': 0.0001}}


for _, d in experiment.items():
    for key, value in d.items():
        hyper[key] = value
    run_sop(hyper=hyper, results_path='./Log/', data=data)

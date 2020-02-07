from Utils.load_data import load_mnist_sop_data
from Models.SOPOptimizer import run_sop
import tensorflow as tf

model_type = 'IGR_I'
hyper = {'width_height': (14, 28, 1),
         'model_type': model_type,
         'batch_size': 64, 'learning_rate': 0.0003,
         'epochs': 100, 'iter_per_epoch': 937, 'temp': tf.constant(0.5)}
data = load_mnist_sop_data(batch_n=hyper['batch_size'], epochs=hyper['epochs'])
run_sop(hyper=hyper, results_path='./Log/', data=data)

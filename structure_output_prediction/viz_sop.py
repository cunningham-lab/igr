from Utils.load_data import load_mnist_sop_data
from Models.SOP import SOP
from Models.SOPOptimizer import viz_reconstruction
import tensorflow as tf

model_type = 'GS'
hyper = {'width_height': (14, 28, 1),
         'model_type': model_type,
         'batch_size': 64, 'learning_rate': 0.0003,
         'epochs': 100, 'iter_per_epoch': 937, 'temp': tf.constant(0.67)}
data = load_mnist_sop_data(batch_n=hyper['batch_size'], epochs=hyper['epochs'])
train, test = data
model = SOP(hyper=hyper)
results_file = './Log/model_weights_GS.h5'
shape = (hyper['batch_size'],) + hyper['width_height']
shape = (64, 14, 28, 1)
model.build(input_shape=shape)
model.load_weights(filepath=results_file)
for x_test in test.take(10):
    images = x_test

viz_reconstruction(test_image=images, model=model)

from Models.SOPOptimizer import run_sop_for_all_cases
import tensorflow as tf

model_type = 'IGR_I'
# model_type = 'IGR_Planar'
# model_type = 'IGR_SB'
# model_type = 'GS'
temp = 1.0

# import numpy as np
# np.random.seed(21)
# seeds = np.random.randint(low=1, high=int(1.e4), size=5)
seeds = [5328, 5945, 8965, 49, 9337]
# seeds = [5328]
architectures = ['double_linear']
sample_sizes = [1]
# architectures = ['double_linear', 'triple_linear', 'nonlinear']
# sample_sizes = [1, 5, 50]
baseline_hyper = {'width_height': (14, 28, 1),
                  'units_per_layer': 240,
                  'model_type': model_type,
                  'batch_size': 100,
                  'learning_rate': 0.001,
                  'weight_decay': 1.e-3,
                  'min_learning_rate': 1.e-4,
                  'epochs': 1 * int(1.e2),
                  # 'check_every': 50,
                  'check_every': 50,
                  'iter_per_epoch': 937,
                  'test_sample_size': 1 * int(1.e2),
                  'architecture': 'nonlinear',
                  'sample_size': 1,
                  'temp': tf.constant(temp)}
idx = 0
variant_hyper = {}
for arch in architectures:
    for sample_size in sample_sizes:
        variant_hyper.update({idx: {'sample_size': sample_size, 'architecture': arch}})
        idx += 1
run_sop_for_all_cases(baseline_hyper, variant_hyper, seeds)

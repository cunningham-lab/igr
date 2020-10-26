import time
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from Models.SOP import SOP, brodcast_to_sample_size
from Utils.load_data import load_mnist_sop_data
from Utils.general import setup_logger
from Utils.general import append_timestamp_to_file


class SOPOptimizer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @tf.function()
    def perform_fwd_pass(self, x_upper, use_one_hot, sample_size=1):
        logits = self.model.call(x_upper=x_upper, use_one_hot=use_one_hot, sample_size=sample_size)
        return logits

    @tf.function()
    def compute_gradients_and_loss(self, x_upper, x_lower, sample_size):
        with tf.GradientTape() as tape:
            logits = self.perform_fwd_pass(x_upper=x_upper, use_one_hot=False,
                                           sample_size=sample_size)
            loss = compute_loss(x_lower=x_lower, logits=logits, sample_size=sample_size)
        gradients = tape.gradient(target=loss, sources=self.model.trainable_variables)
        return gradients, loss

    @tf.function()
    def compute_loss_for_testing(self, x_upper, x_lower, use_one_hot, sample_size):
        logits = self.perform_fwd_pass(x_upper=x_upper,
                                       use_one_hot=use_one_hot, sample_size=sample_size)
        loss = compute_loss(x_lower=x_lower, logits=logits, sample_size=sample_size)
        return loss

    @tf.function()
    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


@tf.function()
def compute_loss(x_lower, logits, sample_size):
    width_loc, height_loc, rgb_loc = 1, 2, 3
    x_lower_broad = brodcast_to_sample_size(x_lower, sample_size)
    log_pxl_z_broad = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_lower_broad, logits=logits)
    log_pxl_z = tf.math.reduce_sum(log_pxl_z_broad, axis=[width_loc, height_loc, rgb_loc])
    loss = -tf.math.reduce_logsumexp(log_pxl_z, axis=1)
    loss = tf.math.reduce_mean(loss, axis=0)
    loss += tf.math.log(tf.constant(sample_size, dtype=tf.float32))
    return loss


def run_sop(hyper, results_path):
    tf.random.set_seed(seed=hyper['seed'])
    data = load_mnist_sop_data(batch_n=hyper['batch_size'])
    train_dataset, test_dataset = data

    sop_optimizer = setup_sop_optimizer(hyper=hyper)

    model_type = sop_optimizer.model.model_type
    log_path = results_path + f'/loss_{model_type}.log'
    logger = setup_logger(log_file_name=append_timestamp_to_file(file_name=log_path,
                                                                 termination='.log'),
                          logger_name=model_type + str(hyper['seed']))
    log_all_hyperparameters(hyper=hyper, logger=logger)
    save_hyper(hyper)
    train_sop(sop_optimizer=sop_optimizer, hyper=hyper, train_dataset=train_dataset,
              test_dataset=test_dataset, logger=logger)


def setup_sop_optimizer(hyper):
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'],
                                         beta_1=0.9,
                                         beta_2=0.999)
    model = SOP(hyper=hyper)
    sop_optimizer = SOPOptimizer(model=model, optimizer=optimizer)
    return sop_optimizer


def log_all_hyperparameters(hyper, logger):
    for key, value in hyper.items():
        logger.info(f'Hyper: {key}: {value}')


def train_sop(sop_optimizer, hyper, train_dataset, test_dataset, logger):
    initial_time = time.time()
    iteration_counter = 0
    sample_size = hyper['sample_size']
    for epoch in range(1, hyper['epochs'] + 1):
        train_mean_loss = tf.keras.metrics.Mean()
        tic = time.time()
        for x_train in train_dataset.take(hyper['iter_per_epoch']):
            x_train_lower = x_train[:, 14:, :, :]
            x_train_upper = x_train[:, :14, :, :]

            gradients, loss = sop_optimizer.compute_gradients_and_loss(x_upper=x_train_upper,
                                                                       x_lower=x_train_lower,
                                                                       sample_size=sample_size)
            sop_optimizer.apply_gradients(gradients=gradients)
            update_learning_rate(sop_optimizer, epoch, iteration_counter, hyper)
            train_mean_loss(loss)
            iteration_counter += 1
        time_taken = time.time() - tic
        if epoch % hyper['check_every'] == 0 or epoch == hyper['epochs']:
            evaluate_progress(epoch=epoch, sop_optimizer=sop_optimizer,
                              test_dataset=test_dataset,
                              train_dataset=train_dataset,
                              logger=logger, hyper=hyper,
                              train_mean_loss=train_mean_loss,
                              iteration_counter=iteration_counter,
                              tic=tic)
        else:
            logger.info(f'Epoch {epoch:4d} || Test_Recon 0.00000e+00 || '
                        f'Train_Recon {train_mean_loss.result().numpy():2.3e} || '
                        f'Temp {sop_optimizer.model.temp:2.1e} || '
                        f'{sop_optimizer.model.model_type} || '
                        f'{sop_optimizer.optimizer.learning_rate.numpy():1.1e} || '
                        f'Time {time_taken:4.1f} sec')

    final_time = time.time()
    logger.info(f'Total training time {final_time - initial_time: 4.1f} secs')
    results_file = f'./Log/model_weights_{sop_optimizer.model.model_type}.h5'
    results_file = append_timestamp_to_file(file_name=results_file, termination='.h5')
    sop_optimizer.model.save_weights(filepath=results_file)


def update_learning_rate(sop_optimizer, epoch, iteration_counter, hyper):
    lr = get_learning_rate_from_scheduler(sop_optimizer, epoch, iteration_counter, hyper)
    sop_optimizer.optimizer.learning_rate = lr


def get_learning_rate_from_scheduler(sop_optimizer, epoch, iteration_counter, hyper):
    lr = sop_optimizer.optimizer.learning_rate.numpy()
    lr = hyper['learning_rate'] * (1 / (1 + hyper['weight_decay'] * iteration_counter))
    lr = np.max((hyper['min_learning_rate'], lr))
    lr = tf.constant(lr, dtype=tf.float32)
    return lr


def evaluate_progress(epoch, sop_optimizer, test_dataset, train_dataset, train_mean_loss,
                      logger, hyper, iteration_counter, tic):
    test_mean_loss = evaluate_loss_on_dataset(test_dataset, sop_optimizer, hyper)
    if epoch == hyper['epochs']:
        train_mean_loss = evaluate_loss_on_dataset(train_dataset, sop_optimizer, hyper)
    lr = sop_optimizer.optimizer.learning_rate.numpy()
    time_taken = time.time() - tic
    logger.info(f'Epoch {epoch:4d} || Test_Recon {test_mean_loss.result().numpy():2.5e} || '
                f'Train_Recon {train_mean_loss.result().numpy():2.3e} || '
                f'Temp {sop_optimizer.model.temp:2.1e} || '
                f'{sop_optimizer.model.model_type} || '
                f'{lr:1.1e} || '
                f'Time {time_taken:4.1f} sec')


def evaluate_loss_on_dataset(dataset, sop_optimizer, hyper):
    mean_loss = tf.keras.metrics.Mean()
    for x in dataset.take(hyper['iter_per_epoch']):
        loss = evaluate_loss_on_batch(x, sop_optimizer, hyper)
        mean_loss(loss)
    return mean_loss


def evaluate_loss_on_batch(x, sop_optimizer, hyper):
    x_lower = x[:, 14:, :, :]
    x_upper = x[:, :14, :, :]
    loss = sop_optimizer.compute_loss_for_testing(x_upper=x_upper,
                                                  x_lower=x_lower, use_one_hot=True,
                                                  sample_size=hyper['test_sample_size'])
    return loss


def run_sop_for_all_cases(baseline_hyper, variant_hyper, seeds):
    for _, variant in variant_hyper.items():
        hyper_copy = dict(baseline_hyper)
        hyper_copy = fill_in_dict(hyper_copy, variant)

        for seed in seeds:
            hyper_copy['seed'] = seed
            run_sop(hyper=hyper_copy, results_path='./Log/')


def save_hyper(hyper):
    model_name = hyper['model_type']
    location = './Log/hyper_' + model_name + '.pkl'
    with open(file=location, mode='wb') as f:
        pickle.dump(hyper, f)


def fill_in_dict(hyper, cases):
    for k, v in cases.items():
        hyper[k] = v
    return hyper


def viz_reconstruction(test_image, model):
    x_test_upper = test_image[:, :14, :, :]
    logits = model.call(x_upper=x_test_upper)
    reconstruction = tf.math.round(tf.math.sigmoid(logits))
    image = np.concatenate((x_test_upper.numpy(), reconstruction), axis=1)
    selected_list = [10, 23, 2, 5, 20, 11, 0, 12, 33, 52]
    for selected in selected_list:
        plt.figure()
        plt.imshow(image[selected, :, :, 0])
        plt.savefig(f'./Results/recon_{selected}.png')

        plt.figure()
        plt.imshow(test_image.numpy()[selected, :, :, 0], cmap='gray')
        plt.savefig(f'./Results/recon_original_{selected}.png')

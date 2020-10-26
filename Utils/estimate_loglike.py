import os
import logging
import time
import pickle
import tensorflow as tf
from Models.train_vae import construct_nets_and_optimizer
from Utils.load_data import load_vae_dataset


def manage_files(path):
    weights_file = 'vae.h5'
    check_hyper = os.path.isfile(path + 'hyper.pkl')
    check_selected_vae = os.path.isfile(path + weights_file)
    if not check_selected_vae:
        w_files = [f for f in os.listdir(path) if f.endswith('.h5')]
        weights_file = w_files[-1]
    checks = (check_hyper, check_selected_vae)
    return checks, weights_file


def get_available_logs(path):
    dirs_available = [d for d in os.listdir(path)
                      if os.path.isdir(os.path.join(path, d))]
    return dirs_available


def estimate_log_likelihood(path_to_trained_models, dataset_name, weights_file, logger,
                            samples_n, model_type, run_with_sample):
    tic = time.time()
    test_dataset, hyper, epoch = load_hyper_and_data(path_to_trained_models,
                                                     dataset_name,
                                                     samples_n, run_with_sample)
    vae_opt = setup_optimizer(path_to_trained_models, hyper, model_type, weights_file)
    calculate_test_log_likelihood(logger, vae_opt, test_dataset, epoch, model_type, tic)


def load_hyper_and_data(path_to_trained_models, dataset_name,
                        samples_n, run_with_sample):
    hyper = load_hyper(path_to_trained_models, samples_n)
    tf.random.set_seed(seed=hyper['seed'])
    data = load_vae_dataset(dataset_name=dataset_name, batch_n=hyper['batch_n'],
                            epochs=hyper['epochs'],
                            run_with_sample=run_with_sample,
                            architecture=hyper['architecture'], hyper=hyper)
    # train_dataset, test_dataset, _, hyper = data
    _, test_dataset, _, hyper = data
    epoch = hyper['epochs']
    return test_dataset, hyper, epoch
    # return train_dataset, hyper, epoch


def load_hyper(path_to_trained_models, samples_n, hyper_file='hyper.pkl'):
    with open(file=path_to_trained_models + hyper_file, mode='rb') as f:
        hyper = pickle.load(f)
    hyper['sample_size_testing'] = samples_n
    hyper['sample_from_disc_kl'] = False
    if 'dtype' not in hyper.keys():
        hyper['dtype'] = tf.float32
    return hyper


def setup_optimizer(path_to_trained_models, hyper, model_type, weights_file='vae.h5'):
    vae_opt = construct_nets_and_optimizer(hyper=hyper, model_type=model_type)
    # TODO: check how to implement better the logic below
    try:
        vae_opt.nets(hyper['image_shape'])
    except Exception:
        vae_opt.nets.load_weights(filepath=path_to_trained_models + weights_file)
        vae_opt.test_with_one_hot = True
        vae_opt.run_iwae = True
    return vae_opt


def calculate_test_log_likelihood(logger, vae_opt, test_dataset, epoch, model_type, tic):
    test_loss_mean = tf.keras.metrics.Mean()
    for x_test in test_dataset:
        loss = vae_opt.compute_losses_from_x_wo_gradients(x_test,
                                                          sample_from_cont_kl=False,
                                                          sample_from_disc_kl=False)
        test_loss_mean(loss)
    evaluation_print = f'Epoch {epoch:4d} || '
    evaluation_print += f'TeELBOC {-test_loss_mean.result():2.5e} || '
    evaluation_print += f'{model_type} || '
    toc = time.time()
    evaluation_print += f'Time: {toc - tic:2.2e} sec'
    logger.info(evaluation_print)


def setup_logger(log_file_name, logger_name: str = None):
    if logger_name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:    %(message)s')
    stream_formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(filename=log_file_name)
    file_handler.setFormatter(fmt=formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=stream_formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=stream_handler)
    logger.propagate = False
    return logger

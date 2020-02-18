import time
import os
import pickle
import tensorflow as tf
from Utils.load_data import load_vae_dataset
from Models.VAENet import construct_networks, determine_path_to_save_results
from Models.OptVAE import OptVAE, OptIGR, OptSB, OptSBFinite, OptExpGS
from Models.OptVAE import OptIGRDis, OptExpGSDis, OptPlanarNFDis, OptPlanarNF
from Utils.viz_vae import plot_originals, plot_reconstructions_samples_and_traversals
from Utils.general import setup_logger, append_timestamp_to_file


# Train VAE
# ===========================================================================================================
def run_vae(hyper, run_with_sample):
    data = load_vae_dataset(dataset_name=hyper['dataset_name'], batch_n=hyper['batch_n'],
                            epochs=hyper['epochs'], run_with_sample=run_with_sample,
                            architecture=hyper['architecture'], hyper=hyper)
    train_dataset, test_dataset, test_images, hyper = data

    vae_opt = construct_nets_and_optimizer(hyper=hyper, model_type=hyper['model_type'])

    train_vae(vae_opt=vae_opt, hyper=hyper, train_dataset=train_dataset,
              test_dataset=test_dataset, test_images=test_images, check_every=hyper['check_every'])


def construct_nets_and_optimizer(hyper, model_type):
    nets = construct_networks(hyper=hyper)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'])
    if model_type == 'VAE':
        vae_opt = OptVAE(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS':
        vae_opt = OptExpGS(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS_Dis':
        vae_opt = OptExpGSDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_I':
        vae_opt = OptIGR(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_I_Dis':
        vae_opt = OptIGRDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_SB':
        vae_opt = OptSB(nets=nets, optimizer=optimizer, hyper=hyper, use_continuous=True)
    elif model_type == 'IGR_SB_Finite':
        vae_opt = OptSBFinite(nets=nets, optimizer=optimizer, hyper=hyper, use_continuous=True)
    elif model_type == 'IGR_SB_Dis':
        vae_opt = OptSB(nets=nets, optimizer=optimizer, hyper=hyper, use_continuous=False)
    elif model_type == 'IGR_SB_Finite_Dis':
        vae_opt = OptSBFinite(nets=nets, optimizer=optimizer, hyper=hyper, use_continuous=False)
    elif model_type == 'IGR_Planar_Dis':
        vae_opt = OptPlanarNFDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_Planar':
        vae_opt = OptPlanarNF(nets=nets, optimizer=optimizer, hyper=hyper)
    else:
        raise RuntimeError
    return vae_opt


def train_vae(vae_opt, hyper, train_dataset, test_dataset, test_images, check_every, monitor_gradients=False):
    logger, results_path = start_all_logging_instruments(hyper=hyper, test_images=test_images)
    init_vars = run_initialization_procedure(hyper, results_path)
    (hyper_file, iteration_counter, results_file, cont_c_linspace, disc_c_linspace, grad_monitor_dict,
     grad_norm) = init_vars

    initial_time = time.time()
    for epoch in range(1, hyper['epochs'] + 1):
        t0 = time.time()
        train_loss_mean = tf.keras.metrics.Mean()
        for x_train in train_dataset.take(hyper['iter_per_epoch']):
            vae_opt, iteration_counter = perform_train_step(x_train, vae_opt, train_loss_mean,
                                                            iteration_counter, disc_c_linspace, cont_c_linspace)
        t1 = time.time()
        # noinspection PyUnboundLocalVariable
        monitor_vanishing_grads(monitor_gradients, x_train, vae_opt,
                                iteration_counter, grad_monitor_dict, epoch)

        evaluate_progress_in_test_set(epoch=epoch, test_dataset=test_dataset, vae_opt=vae_opt,
                                      hyper=hyper, logger=logger, iteration_counter=iteration_counter,
                                      train_loss_mean=train_loss_mean, time_taken=t1 - t0,
                                      check_every=check_every)

        save_intermediate_results(epoch, vae_opt, test_images, hyper, results_file, results_path)

    save_final_results(vae_opt.nets, logger, results_file, initial_time, temp=vae_opt.temp.numpy())


def start_all_logging_instruments(hyper, test_images):
    results_path = determine_path_to_save_results(model_type=hyper['model_type'],
                                                  dataset_name=hyper['dataset_name'])
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    logger = setup_logger(log_file_name=append_timestamp_to_file(file_name=results_path + '/loss.log',
                                                                 termination='.log'),
                          logger_name=append_timestamp_to_file('logger', termination=''))
    log_all_hyperparameters(hyper=hyper, logger=logger)
    plot_originals(test_images=test_images, results_path=results_path)
    return logger, results_path


def log_all_hyperparameters(hyper, logger):
    logger.info(f"GPU Available: {tf.test.is_gpu_available()}")
    for key, value in hyper.items():
        logger.info(f'Hyper: {key}: {value}')


def run_initialization_procedure(hyper, results_path):
    init_vars = initialize_vae_variables(results_path=results_path, hyper=hyper)
    hyper_file, *_ = init_vars

    with open(file=hyper_file, mode='wb') as f:
        pickle.dump(obj=hyper, file=f)

    return init_vars


def initialize_vae_variables(results_path, hyper):
    iteration_counter = 0
    results_file = results_path + '/vae.h5'
    hyper_file = results_path + '/hyper.pkl'
    cont_c_linspace = convert_into_linspace(hyper['cont_c_linspace'])
    disc_c_linspace = convert_into_linspace(hyper['disc_c_linspace'])
    grad_monitor_dict = {}
    grad_norm = tf.constant(0., dtype=tf.float32)
    init_vars = (hyper_file, iteration_counter, results_file, cont_c_linspace, disc_c_linspace, grad_monitor_dict,
                 grad_norm)
    return init_vars


def convert_into_linspace(limits_tuple):
    var_linspace = tf.linspace(start=limits_tuple[0], stop=limits_tuple[1], num=limits_tuple[2])
    return var_linspace


def perform_train_step(x_train, vae_opt, train_loss_mean, iteration_counter, disc_c_linspace, cont_c_linspace):
    output = vae_opt.compute_gradients(x=x_train)
    gradients, loss, log_px_z, kl, kl_n, kl_d = output
    vae_opt.apply_gradients(gradients=gradients)
    iteration_counter += 1
    train_loss_mean(loss)
    update_regularization_channels(vae_opt=vae_opt, iteration_counter=iteration_counter,
                                   disc_c_linspace=disc_c_linspace, cont_c_linspace=cont_c_linspace)
    return vae_opt, iteration_counter


def update_regularization_channels(vae_opt, iteration_counter, disc_c_linspace, cont_c_linspace):
    if iteration_counter < disc_c_linspace.shape[0]:
        vae_opt.continuous_c = cont_c_linspace[iteration_counter]
        vae_opt.discrete_c = disc_c_linspace[iteration_counter]


def monitor_vanishing_grads(monitor_gradients, x_train, vae_opt, iteration_counter, grad_monitor_dict, epoch):
    if monitor_gradients:
        grad_norm = vae_opt.monitor_parameter_gradients_at_psi(x=x_train)
        grad_monitor_dict.update({iteration_counter: grad_norm.numpy()})
        with open(file='./Results/gradients_' + str(epoch) + '.pkl', mode='wb') as f:
            pickle.dump(obj=grad_monitor_dict, file=f)


def evaluate_progress_in_test_set(epoch, test_dataset, vae_opt, hyper, logger, iteration_counter,
                                  time_taken, train_loss_mean, check_every=10):
    if epoch % check_every == 0:
        test_progress = create_test_progress_tracker()
        for x_test in test_dataset.take(hyper['iter_per_epoch']):
            test_progress = update_test_progress(x_test, vae_opt, test_progress)
        log_test_progress(logger, test_progress, epoch, time_taken, iteration_counter, train_loss_mean, vae_opt.temp)


def create_test_progress_tracker():
    vars_to_track = {'TeELBO': (False, False), 'TeELBOC': (False, True), 'TeJV': (True, False),
                     'TeJVC': (True, True), 'N': ()}
    test_track = {'vars_to_track': vars_to_track}
    for k, _ in vars_to_track.items():
        test_track[k] = tf.keras.metrics.Mean()
    return test_track


def update_test_progress(x_test, vae_opt, test_progress):
    test_progress['N'](vae_opt.n_required)
    for k, v in test_progress['vars_to_track'].items():
        if k != 'N':
            loss, *_ = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_jv=v[0], run_closed_form_kl=v[1])
            test_progress[k](loss)
    return test_progress


def log_test_progress(logger, test_progress, epoch, time_taken, iteration_counter, train_loss_mean, temp):
    test_print = f'Epoch {epoch:4d} || '
    for k, _ in test_progress['vars_to_track'].items():
        loss = test_progress[k].result().numpy()
        if k != 'N':
            test_print += f'{k} {-loss:2.5e} || '
        else:
            test_print += f'{k} {int(loss):2d} || '
    test_print += (f'TrL {train_loss_mean.result().numpy():2.5e} || ' +
                   f'{time_taken:4.1f} sec || i: {iteration_counter:6,d} || ')
    logger.info(test_print)
    tf.summary.scalar(name='Test ELBO', data=-test_progress['TeELBO'].result(), step=epoch)
    tf.summary.scalar(name='N Required', data=test_progress['N'].result(), step=epoch)
    tf.summary.scalar(name='Temp', data=temp, step=epoch)


def save_intermediate_results(epoch, vae_opt, test_images, hyper, results_file, results_path, save_every=25):
    if epoch % save_every == 0:
        vae_opt.nets.save_weights(filepath=append_timestamp_to_file(results_file, '.h5'))
        plot_reconstructions_samples_and_traversals(hyper=hyper, epoch=epoch, results_path=results_path,
                                                    test_images=test_images, vae_opt=vae_opt)


def save_final_results(nets, logger, results_file, initial_time, temp):
    final_time = time.time()
    logger.info(f'Total training time {final_time - initial_time: 4.1f} secs')
    logger.info(f'Final temp {temp: 4.5f}')
    results_file = append_timestamp_to_file(file_name=results_file, termination='.h5')
    nets.save_weights(filepath=results_file)


def run_vae_for_all_cases(hyper, model_cases, dataset_cases, temps, num_of_repetitions, run_with_sample):
    for _, model in model_cases.items():
        hyper_copy = dict(hyper)
        hyper_copy = fill_in_dict(hyper_copy, model)
        hyper_copy = fill_model_depending_settings(hyper_copy)
        data_and_temps = add_data_and_temp_cases(dataset_cases, temps)

        for _, d_and_t in data_and_temps.items():
            hyper_copy = fill_in_dict(hyper_copy, d_and_t)
            for rep in range(num_of_repetitions):
                run_vae(hyper=hyper_copy, run_with_sample=run_with_sample)


def fill_in_dict(hyper, cases):
    for k, v in cases.items():
        hyper[k] = v
    return hyper


def fill_model_depending_settings(hyper_copy):
    hyper_copy['latent_discrete_n'] = hyper_copy['n_required']
    if hyper_copy['model_type'].find('GS') >= 0:
        hyper_copy['run_closed_form_kl'] = False
        hyper_copy['num_of_discrete_param'] = 1
    else:
        hyper_copy['latent_discrete_n'] += 1
        hyper_copy['run_closed_form_kl'] = True
        hyper_copy['num_of_discrete_param'] = 2
    return hyper_copy


def add_data_and_temp_cases(dataset_cases, temps):
    data_and_temp = {}
    i = 0
    for _, c in dataset_cases.items():
        for t in temps:
            i += 1
            data_and_temp.update({i: {}})
            c.update({'temp': t})
            for key, val in c.items():
                data_and_temp[i][key] = val
    return data_and_temp

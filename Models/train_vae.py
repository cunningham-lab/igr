import time
import os
import pickle
import tensorflow as tf
from Utils.load_data import load_vae_dataset
from Models.VAENet import construct_networks, determine_path_to_save_results
from Models.OptVAE import OptVAE, OptIGR, OptSB, OptSBFinite
from Models.OptVAE import OptIGRDis, OptExpGSDis, OptPlanarNFDis, OptPlanarNF
from Models.OptVAE import OptRELAXGSDis, OptRELAXBerDis, OptRELAXIGR
from Models.OptVAE import OptDLGMM, OptDLGMMIGR, OptDLGMMIGR_SB, OptDLGMM_Var
from Utils.viz_vae import plot_originals
from Utils.general import setup_logger, append_timestamp_to_file


def run_vae(hyper, run_with_sample):
    tf.random.set_seed(seed=hyper['seed'])
    data = load_vae_dataset(dataset_name=hyper['dataset_name'], batch_n=hyper['batch_n'],
                            epochs=hyper['epochs'], run_with_sample=run_with_sample,
                            architecture=hyper['architecture'], hyper=hyper)
    train_dataset, test_dataset, test_images, hyper = data

    vae_opt = construct_nets_and_optimizer(hyper=hyper, model_type=hyper['model_type'])

    train_vae(vae_opt=vae_opt, hyper=hyper, train_dataset=train_dataset,
              test_dataset=test_dataset, test_images=test_images,
              check_every=hyper['check_every'])


def construct_nets_and_optimizer(hyper, model_type):
    nets = construct_networks(hyper=hyper)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'])
    if model_type == 'VAE':
        vae_opt = OptVAE(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'DLGMM':
        vae_opt = OptDLGMM(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'DLGMM_Var':
        vae_opt = OptDLGMM_Var(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'DLGMM_IGR':
        vae_opt = OptDLGMMIGR(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'DLGMM_IGR_SB':
        vae_opt = OptDLGMMIGR_SB(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS':
        vae_opt = OptExpGSDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS_Dis':
        vae_opt = OptExpGSDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_I':
        vae_opt = OptIGR(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_I_Dis':
        vae_opt = OptIGRDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_SB':
        vae_opt = OptSB(nets=nets, optimizer=optimizer, hyper=hyper, use_continuous=True)
    elif model_type == 'IGR_SB_Finite':
        vae_opt = OptSBFinite(nets=nets, optimizer=optimizer, hyper=hyper,
                              use_continuous=True)
    elif model_type == 'IGR_SB_Dis':
        vae_opt = OptSB(nets=nets, optimizer=optimizer,
                        hyper=hyper, use_continuous=False)
    elif model_type == 'IGR_SB_Finite_Dis':
        vae_opt = OptSBFinite(nets=nets, optimizer=optimizer,
                              hyper=hyper, use_continuous=False)
    elif model_type == 'IGR_Planar_Dis':
        vae_opt = OptPlanarNFDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_Planar':
        vae_opt = OptPlanarNF(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type.find('Relax') >= 0:
        optimizer_decoder = optimizer
        optimizer_encoder = tf.keras.optimizers.Adam(
            learning_rate=hyper['learning_rate'])
        optimizer_var = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'])
        optimizers = (optimizer_encoder, optimizer_decoder, optimizer_var)
        if model_type == 'Relax_GS_Dis':
            vae_opt = OptRELAXGSDis(nets=nets, optimizers=optimizers, hyper=hyper)
        elif model_type == 'Relax_Ber_Dis':
            vae_opt = OptRELAXBerDis(nets=nets, optimizers=optimizers, hyper=hyper)
        elif model_type == 'Relax_IGR':
            vae_opt = OptRELAXIGR(nets=nets, optimizers=optimizers, hyper=hyper)
    else:
        raise RuntimeError
    return vae_opt


def train_vae(vae_opt, hyper, train_dataset, test_dataset, test_images, check_every):
    logger, results_path = start_all_logging_instruments(
        hyper=hyper, test_images=test_images)
    hyper_file, results_file, = run_initialization_procedure(hyper, results_path)

    initial_time = time.time()
    for epoch in range(1, hyper['epochs'] + 1):
        t0 = time.time()
        vae_opt.train_on_epoch(train_dataset, hyper['iter_per_epoch'])
        t1 = time.time()
        log_train_progress(logger, epoch, t1 - t0, vae_opt.iter_count,
                           vae_opt.train_loss_mean, vae_opt.n_required)
        evaluate_progress_in_test_set(epoch=epoch, test_dataset=test_dataset,
                                      vae_opt=vae_opt,
                                      hyper=hyper, logger=logger, time_taken=t1 - t0,
                                      check_every=check_every)
        save_intermediate_results(epoch, vae_opt, test_images,
                                  hyper, results_file, results_path)

    save_final_results(vae_opt.nets, logger, results_file,
                       initial_time, temp=vae_opt.temp.numpy())


def start_all_logging_instruments(hyper, test_images):
    results_path = determine_path_to_save_results(model_type=hyper['model_type'],
                                                  dataset_name=hyper['dataset_name'])
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    logger = setup_logger(log_file_name=append_timestamp_to_file(file_name=results_path +
                                                                 '/loss.log',
                                                                 termination='.log'),
                          logger_name=append_timestamp_to_file('logger', termination=''))
    log_all_hyperparameters(hyper=hyper, logger=logger)
    plot_originals(test_images=test_images, results_path=results_path)
    return logger, results_path


def log_all_hyperparameters(hyper, logger):
    logger.info(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    for key, value in hyper.items():
        logger.info(f'Hyper: {key}: {value}')


def run_initialization_procedure(hyper, results_path):
    results_file = results_path + '/vae.h5'
    hyper_file = results_path + '/hyper.pkl'

    with open(file=hyper_file, mode='wb') as f:
        pickle.dump(obj=hyper, file=f)

    return hyper_file, results_file


def print_gradient_analysis(relax, g2, iteration_counter, loss, other=None):
    relax = relax[0] if len(relax) > 0 else relax
    if len(g2) > 1:
        mu = tf.constant(5.) * tf.math.tanh(g2[0])
        xi = tf.constant(2.) * tf.math.sigmoid(g2[1]) + tf.constant(0.5)
    else:
        mu = g2[0]
        xi = tf.math.exp(g2)[0]
    if iteration_counter % 100 == 0:
        print('\n')
        tf.print((iteration_counter, loss))
        gnorm, gmax, gmean, gmin = get_statistics(relax)
        print(f'Lax:   ({gmin:+1.2e}, {gmean:+1.2e}, {gmax:+1.2e}) -> {gnorm:+1.2e}')
        print('+++++++++')
        gnorm, gmax, gmean, gmin = get_statistics(mu)
        print(f'Mu:    ({gmin:+1.2e}, {gmean:+1.2e}, {gmax:+1.2e}) -> {gnorm:+1.2e}')
        gnorm, gmax, gmean, gmin = get_statistics(xi)
        print(f'Sigma: ({gmin:+1.2e}, {gmean:+1.2e}, {gmax:+1.2e}) -> {gnorm:+1.2e}')
        recon = other.numpy()
        print(f'Recon Loss {recon:+1.3e} || log_qz|x {-recon + loss - 46:+1.3e}')


def get_statistics(g):
    norm = tf.math.sqrt(tf.reduce_sum(g ** 2))
    gmax = tf.reduce_max(g)
    gmean = tf.math.reduce_mean(g)
    gmin = tf.reduce_min(g)
    return norm, gmax, gmean, gmin


def monitor_vanishing_grads(monitor_gradients, x_train, vae_opt, iteration_counter,
                            grad_monitor_dict, epoch):
    if monitor_gradients:
        grad_norm = vae_opt.monitor_parameter_gradients_at_psi(x=x_train)
        grad_monitor_dict.update({iteration_counter: grad_norm.numpy()})
        with open(file='./Results/gradients_' + str(epoch) + '.pkl', mode='wb') as f:
            pickle.dump(obj=grad_monitor_dict, file=f)


def evaluate_progress_in_test_set(epoch, test_dataset, vae_opt, hyper, logger,
                                  time_taken, check_every=10):
    if epoch % check_every == 0 or epoch == 1 or epoch == hyper['epochs']:
        test_progress = create_test_progress_tracker()
        for x_test in test_dataset.take(hyper['iter_per_epoch']):
            test_progress = update_test_progress(x_test, vae_opt, test_progress)
        log_test_progress(logger, test_progress, epoch, time_taken, vae_opt.iter_count,
                          vae_opt.train_loss_mean, vae_opt.temp)


def create_test_progress_tracker():
    vars_to_track = {'TeELBO': True, 'TeELBOC': False, 'N': ()}
    test_track = {'vars_to_track': vars_to_track}
    for k, _ in vars_to_track.items():
        test_track[k] = tf.keras.metrics.Mean()
    return test_track


def update_test_progress(x_test, vae_opt, test_progress):
    test_progress['N'](vae_opt.n_required)
    for k, v in test_progress['vars_to_track'].items():
        if k != 'N':
            loss = vae_opt.compute_losses_from_x_wo_gradients(x=x_test,
                                                              sample_from_cont_kl=v,
                                                              sample_from_disc_kl=v)
            test_progress[k](loss)
    return test_progress


def log_test_progress(logger, test_progress, epoch, time_taken,
                      iteration_counter, train_loss_mean, temp):
    print_msg = f'Epoch {epoch:4d} || '
    for k, _ in test_progress['vars_to_track'].items():
        loss = test_progress[k].result().numpy()
        print_msg += f'{k} {-loss:2.5e} || ' if k != 'N' else f'{k} {int(loss):2d} || '
    print_msg += (f'TrL {train_loss_mean.result().numpy():2.5e} || ' +
                  f'{time_taken:4.1f} sec || i: {iteration_counter:6,d} || ')
    logger.info(print_msg)
    var_name = list(test_progress['vars_to_track'].keys())[0]
    tf.summary.scalar(name='Test ELBO', data=-
                      test_progress[var_name].result(), step=epoch)
    tf.summary.scalar(name='N Required', data=test_progress['N'].result(), step=epoch)
    tf.summary.scalar(name='Temp', data=temp, step=epoch)


def log_train_progress(logger, epoch, time_taken, iteration_counter,
                       train_loss_mean, n_required):
    print_msg = f'Epoch {epoch:4d} || '
    print_msg += (f'TrL {train_loss_mean.result().numpy():2.5e} || ' +
                  f'{time_taken:4.1f} sec || i: {iteration_counter:6,d} || ' +
                  f'N: {n_required: 3d}')
    logger.info(print_msg)


def save_intermediate_results(epoch, vae_opt, test_images, hyper,
                              results_file, results_path):
    if epoch % hyper['save_every'] == 0:
        vae_opt.nets.save_weights(filepath=append_timestamp_to_file(results_file, '.h5'))


def save_final_results(nets, logger, results_file, initial_time, temp):
    final_time = time.time()
    logger.info(f'Total training time {final_time - initial_time: 4.1f} secs')
    logger.info(f'Final temp {temp: 4.5f}')
    results_file = append_timestamp_to_file(file_name=results_file, termination='.h5')
    nets.save_weights(filepath=results_file)


def run_vae_for_all_cases(hyper, model_cases, dataset_cases, temps, seeds,
                          run_with_sample):
    for _, model in model_cases.items():
        hyper_copy = dict(hyper)
        hyper_copy = fill_in_dict(hyper_copy, model)
        hyper_copy = fill_model_depending_settings(hyper_copy)
        data_and_temps = add_data_and_temp_cases(dataset_cases, temps)

        for _, d_and_t in data_and_temps.items():
            hyper_copy = fill_in_dict(hyper_copy, d_and_t)
            for seed in seeds:
                hyper_copy['seed'] = seed
                run_vae(hyper=hyper_copy, run_with_sample=run_with_sample)


def fill_in_dict(hyper, cases):
    for k, v in cases.items():
        hyper[k] = v
    return hyper


def fill_model_depending_settings(hyper_copy):
    hyper_copy['latent_discrete_n'] = hyper_copy['n_required']
    cond = (hyper_copy['model_type'].find('GS') >= 0 or
            hyper_copy['model_type'].find('Ber') >= 0)
    if cond:
        hyper_copy['num_of_discrete_param'] = 1
    else:
        hyper_copy['latent_discrete_n'] += 1
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

import time
import pickle
import tensorflow as tf
from Utils.load_data import load_vae_dataset
from Models.VAENet import setup_model, determine_path_to_save_results
from Models.OptVAE import OptVAE, OptGauSoftMax, OptGS, OptSBVAE, OptExpGS, OptGauSoftPlus
from Models.OptVAE import OptGauSoftMaxDis, OptGSDis, OptIsoGauSoftMaxDis, OptGauSoftPlusDis, OptExpGSDis
from Models.OptVAE import OptPlanarNFDis
from Utils.viz_vae import plot_grid_of_generated_digits, plot_originals
from Utils.viz_vae import plot_reconstructions_samples_and_traversals
from Utils.general import setup_logger
from Utils.general import append_timestamp_to_file
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Train VAE
# ===========================================================================================================


def run_vae(hyper, run_with_sample):
    data = load_vae_dataset(dataset_name=hyper['dataset_name'], batch_n=hyper['batch_n'],
                            epochs=hyper['epochs'], run_with_sample=run_with_sample,
                            architecture=hyper['architecture'])
    (train_dataset, test_dataset, test_images, hyper['batch_n'], hyper['epochs'],
     image_size, hyper['iter_per_epoch']) = data

    results_path = determine_path_to_save_results(model_type=hyper['model_type'],
                                                  dataset_name=hyper['dataset_name'])
    model = setup_model(hyper=hyper, image_size=image_size)

    vae_opt = setup_vae_optimizer(model=model, hyper=hyper, model_type=hyper['model_type'])

    writer, logger = start_all_logging_instruments(hyper=hyper, results_path=results_path,
                                                   test_images=test_images)

    train_vae_model(vae_opt=vae_opt, model=model, writer=writer, hyper=hyper, train_dataset=train_dataset,
                    test_dataset=test_dataset, logger=logger, results_path=results_path,
                    test_images=test_images)

    plot_grid_of_generated_digits(model=model, n_required=vae_opt.n_required, fig_size=vae_opt.n_required,
                                  filename=results_path + '/samples_grid.html', digit_size=image_size[0])


def start_all_logging_instruments(hyper, results_path, test_images):
    writer = tf.summary.create_file_writer(logdir=results_path)
    logger = setup_logger(log_file_name=append_timestamp_to_file(file_name=results_path + '/loss.log',
                                                                 termination='.log'),
                          logger_name=append_timestamp_to_file('logger', termination=''))
    log_all_hyperparameters(hyper=hyper, logger=logger)
    plot_originals(test_images=test_images, results_path=results_path)
    return writer, logger


def log_all_hyperparameters(hyper, logger):
    logger.info(f"GPU Available: {tf.test.is_gpu_available()}")
    for key, value in hyper.items():
        logger.info(f'Hyper: {key}: {value}')


def setup_vae_optimizer(model, hyper, model_type):
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'])
    if model_type == 'VAE':
        vae_opt = OptVAE(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS':
        vae_opt = OptGS(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GSDis':
        vae_opt = OptGSDis(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'ExpGS':
        vae_opt = OptExpGS(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'ExpGSDis':
        vae_opt = OptExpGSDis(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GSM':
        vae_opt = OptGauSoftMax(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GSMDis':
        vae_opt = OptGauSoftMaxDis(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGSMDis':
        vae_opt = OptIsoGauSoftMaxDis(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GSP':
        vae_opt = OptGauSoftPlus(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GSPDis':
        vae_opt = OptGauSoftPlusDis(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'SB':
        vae_opt = OptSBVAE(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'PlanarDis':
        vae_opt = OptPlanarNFDis(model=model, optimizer=optimizer, hyper=hyper)
    else:
        raise RuntimeError
    return vae_opt


def train_vae_model(vae_opt, model, writer, hyper, train_dataset, test_dataset, logger, results_path,
                    test_images, monitor_gradients=False):

    (iteration_counter, results_file, hyper_file,
     cont_c_linspace, disc_c_linspace) = initialize_vae_variables(results_path=results_path, hyper=hyper)
    grad_monitor_dict = {}
    grad_norm = tf.constant(0., dtype=tf.float32)
    with open(file=hyper_file, mode='wb') as f:
        pickle.dump(obj=hyper, file=f)

    with writer.as_default():
        initial_time = time.time()
        for epoch in range(1, hyper['epochs'] + 1):
            t0 = time.time()
            train_loss_mean = tf.keras.metrics.Mean()
            for x_train in train_dataset.take(hyper['iter_per_epoch']):
                output = vae_opt.compute_gradients(x=x_train)
                gradients, loss, log_px_z, kl, kl_n, kl_d = output
                vae_opt.apply_gradients(gradients=gradients)
                iteration_counter += 1
                train_loss_mean(loss)
                append_train_summaries(tracking_losses=output, iteration_counter=iteration_counter)
                update_regularization_channels(vae_opt=vae_opt, iteration_counter=iteration_counter,
                                               disc_c_linspace=disc_c_linspace,
                                               cont_c_linspace=cont_c_linspace)

            if monitor_gradients:
                grad_norm = vae_opt.monitor_parameter_gradients_at_psi(x=x_train)
                grad_monitor_dict.update({iteration_counter: grad_norm.numpy()})
                with open(file='./Results/gradients_' + str(epoch) + '.pkl', mode='wb') as f:
                    pickle.dump(obj=grad_monitor_dict, file=f)

            t1 = time.time()

            evaluate_progress_in_test_set(epoch=epoch, test_dataset=test_dataset, vae_opt=vae_opt,
                                          hyper=hyper, logger=logger, iteration_counter=iteration_counter,
                                          train_loss_mean=train_loss_mean, time_taken=t1 - t0,
                                          grad_norm=grad_norm)

            if epoch % 10 == 0:
                model.save_weights(filepath=append_timestamp_to_file(results_file, '.h5'))
                plot_reconstructions_samples_and_traversals(model=model, hyper=hyper, epoch=epoch,
                                                            results_path=results_path,
                                                            test_images=test_images, vae_opt=vae_opt)
            writer.flush()

        final_time = time.time()
        logger.info(f'Total training time {final_time - initial_time: 4.1f} secs')
        logger.info(f'Final temp {vae_opt.temp.numpy(): 4.5f}')
        results_file = append_timestamp_to_file(file_name=results_file, termination='.h5')
        model.save_weights(filepath=results_file)


def initialize_vae_variables(results_path, hyper):
    iteration_counter = 0
    results_file = results_path + '/vae.h5'
    hyper_file = results_path + '/hyper.pkl'
    continuous_channel_lin = convert_into_linspace(hyper['continuous_c_linspace'])
    discrete_channel_lin = convert_into_linspace(hyper['discrete_c_linspace'])
    return iteration_counter, results_file, hyper_file, continuous_channel_lin, discrete_channel_lin


def convert_into_linspace(limits_tuple):
    var_linspace = tf.linspace(start=limits_tuple[0], stop=limits_tuple[1], num=limits_tuple[2])
    return var_linspace


def update_regularization_channels(vae_opt, iteration_counter, disc_c_linspace, cont_c_linspace):
    if iteration_counter < disc_c_linspace.shape[0]:
        vae_opt.continuous_c = cont_c_linspace[iteration_counter]
        vae_opt.discrete_c = disc_c_linspace[iteration_counter]


def append_train_summaries(tracking_losses, iteration_counter):
    gradients, loss, log_px_z, kl, kl_n, kl_d = tracking_losses
    tf.summary.scalar(name='Train ELBO', data=-loss, step=iteration_counter)
    tf.summary.scalar(name='Train Recon', data=log_px_z, step=iteration_counter)
    tf.summary.scalar(name='Train KL', data=kl, step=iteration_counter)
    tf.summary.scalar(name='Train KL Norm', data=kl_n, step=iteration_counter)
    tf.summary.scalar(name='Train KL Dis', data=kl_d, step=iteration_counter)


def evaluate_progress_in_test_set(epoch, test_dataset, vae_opt, hyper, logger, iteration_counter,
                                  time_taken, train_loss_mean, grad_norm):
    n_mean = tf.keras.metrics.Mean()
    recon_mean, kl_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    kl_n_mean, kl_d_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    jv_closed, jv = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    elbo_closed, elbo = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    use_analytical = hyper['use_analytical_in_test']
    for x_test in test_dataset.take(hyper['iter_per_epoch']):
        jv_closed_loss, *_ = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_ccβvae=True,
                                                                        run_analytical_kl=True)
        jv_loss, *_ = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_ccβvae=True,
                                                                 run_analytical_kl=False)
        output_closed = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_ccβvae=False,
                                                                   run_analytical_kl=True)
        output = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_ccβvae=False,
                                                            run_analytical_kl=False)
        elbo_loss, log_px_z, kl, kl_n, kl_d = output
        elbo_closed_loss, log_px_z_closed, kl_closed, kl_n_closed, kl_d_closed = output_closed
        jv_closed(jv_closed_loss)
        jv(jv_loss)
        elbo(elbo_loss)
        elbo_closed(elbo_closed_loss)
        n_mean(vae_opt.n_required)
        recon_mean(log_px_z)
        if use_analytical:
            kl_mean(kl_closed)
            kl_n_mean(kl_n_closed)
            kl_d_mean(kl_d_closed)
        else:
            kl_mean(kl)
            kl_n_mean(kl_n)
            kl_d_mean(kl_d)
    logger.info(f'Epoch {epoch:4d} || TeELBO {-elbo.result().numpy():2.5e} || '
                f'TeELBOC {-elbo_closed.result().numpy():2.5e} || '
                f'TeJV {jv.result().numpy():2.5e} || '
                f'TeJVC {jv_closed.result().numpy():2.5e} || '
                f'TrL {train_loss_mean.result().numpy():2.5e} || '
                f'{time_taken:4.1f} sec || i: {iteration_counter:6,d} || '
                f'N: {n_mean.result():4.1f}')
    if use_analytical:
        tf.summary.scalar(name='Test ELBO', data=-elbo_closed.result(), step=epoch)
    else:
        tf.summary.scalar(name='Test ELBO', data=-elbo.result(), step=epoch)
    tf.summary.scalar(name='N Required', data=n_mean.result(), step=epoch)
    tf.summary.scalar(name='Temp', data=vae_opt.temp, step=epoch)
    tf.summary.scalar(name='Test Recon', data=recon_mean.result(), step=epoch)
    tf.summary.scalar(name='Test KL', data=kl_mean.result(), step=epoch)
    tf.summary.scalar(name='Test KL Norm', data=kl_n_mean.result(), step=epoch)
    tf.summary.scalar(name='Test KL Dis', data=kl_d_mean.result(), step=epoch)

# ===========================================================================================================

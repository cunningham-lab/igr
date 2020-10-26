from Utils.estimate_loglike import estimate_log_likelihood
from Utils.estimate_loglike import get_available_logs
from Utils.estimate_loglike import manage_files
from Utils.estimate_loglike import setup_logger

run_with_sample = False
check_only = False
samples_n = 1 * int(1.e2)
# datasets = ['mnist', 'fmnist', 'omniglot']
# datasets = ['fmnist', 'omniglot']
# datasets = ['mnist']
# datasets = ['fmnist']
datasets = ['omniglot']
# architectures = ['linear_no_sl']
# architectures = ['linear_w_sl']
# architectures = ['nonlinear_wo_sl']
# architectures = ['nonlinear_w_sl']
architectures = ['nonlinear_w_sl', 'nonlinear_wo_sl']
models = {
    # 1: {'model_dir': 'igr', 'model_type': 'IGR_I_Dis'},
    # 2: {'model_dir': 'pf', 'model_type': 'IGR_Planar_Dis'},
    # 3: {'model_dir': 'sb', 'model_type': 'IGR_SB_Finite_Dis'},
    4: {'model_dir': 'gs', 'model_type': 'GS_Dis'},
    # 5: {'model_dir': 'relax_igr', 'model_type': 'Relax_IGR'},
    # 6: {'model_dir': 'relax_gs', 'model_type': 'Relax_GS_Dis'},
}
logger = setup_logger(log_file_name='./Log/nll.txt', logger_name='nll')
for dataset in datasets:
    for arch in architectures:
        for _, v in models.items():
            path_to_trained_models = './Results/trained_models/' + dataset + '/'
            # path_to_trained_models = './Results/rebuttal/' + dataset + '/'
            path_to_trained_models += v['model_dir'] + '/'
            path_to_trained_models += arch + '/'
            logs = get_available_logs(path_to_trained_models)
            for log in logs:
                current_path = path_to_trained_models + log + '/'
                checks, weights_file = manage_files(path_to_trained_models + log + '/')
                logger.info(current_path)
                logger.info(f'Hyper present: {checks[0]}, Selected weights: {checks[1]}')
                if not check_only:
                    estimate_log_likelihood(current_path, dataset, weights_file, logger,
                                            samples_n, v['model_type'], run_with_sample)

import datetime
import numpy as np
import logging
import tensorflow as tf
from matplotlib import pyplot as plt


def count_zeros_in_gradient(grad_dict):
    grad_np = np.zeros(shape=100)
    i = 0
    for k, v in grad_dict.items():
        # z = np.mean(np.mean(v[:, 0, :], axis=0))
        # z = np.mean(np.var(v[:, 0, :], axis=0))
        # grad_np[i] = z
        v = v.flatten()
        # z = v == 0.
        z = np.abs(v) <= 1.e-10
        grad_np[i] = np.mean(z)
        i += 1
    return grad_np


def add_mean_std_plot_line(runs, color, label, offset=5, linestyle='-'):
    shrinked_runs = runs[:, offset:]
    add_mean_lines(shrinked_runs, label=label, color=color, offset=offset, linestyle=linestyle)
    add_std_lines(shrinked_runs, color=color, offset=offset)


def add_mean_lines(runs, color, offset, label, linestyle):
    run_axis = 0
    runs_num = np.arange(runs.shape[1]) + offset
    run_mean = np.mean(runs, axis=run_axis)
    plt.plot(runs_num, run_mean, label=label, color=color, linestyle=linestyle)


def add_std_lines(runs, color, offset, alpha=0.5):
    run_axis = 0
    runs_num = np.arange(runs.shape[1]) + offset
    run_mean = np.mean(runs, axis=run_axis)
    total_runs = runs.shape[run_axis]
    run_std = np.std(runs, axis=run_axis) / total_runs
    plt.vlines(runs_num, ymin=run_mean - run_std, ymax=run_mean + run_std, color=color, alpha=alpha)


def make_np_of_var_from_log_files(variable_name: str, files_list: list, path_to_files: str):
    results_list = []
    for f in files_list:
        if not f.startswith('.'):
            variable_np = get_variable_np_array_from_log_file(variable_name=variable_name,
                                                              path_to_file=path_to_files + f)
            results_list.append(variable_np)
    results_np = create_global_np_array_from_results(results_list=results_list)
    return results_np


def get_variable_np_array_from_log_file(variable_name: str, path_to_file: str):
    variable_results = []
    with open(file=path_to_file, mode='r') as f:
        lines = f.readlines()
        for l in lines:
            split = l.split(sep='||')
            if len(split) > 1:
                for part in split:
                    if part.find(variable_name) > 0:
                        var = float(part.split()[1])
                        variable_results.append(var)
        variable_np = np.array(variable_results)
    return variable_np


def create_global_np_array_from_results(results_list: list):
    total_runs = len(results_list)
    size_of_run = results_list[0].shape[0]
    results_np = np.zeros(shape=(total_runs, size_of_run))
    for run in range(total_runs):
        results_np[run, :] = results_list[run]
    return results_np


def reshape_parameter_for_model(shape, param):
    batch_n, categories_n, sample_size, num_of_vars = shape
    param = np.reshape(param, newshape=(batch_n, categories_n, 1, 1))
    param = np.broadcast_to(param, shape=(batch_n, categories_n, sample_size, 1))
    param = np.broadcast_to(param, shape=(batch_n, categories_n, sample_size, num_of_vars))
    param = tf.constant(param, dtype=tf.float32)
    return param


def convert_into_one_hot(shape, categorical):
    batch_n, categories_n, sample_size, num_of_vars = shape
    categorical_one_hot = np.zeros(shape=shape)
    for i in range(sample_size):
        for j in range(num_of_vars):
            max_i = categorical[0, i, j]
            categorical_one_hot[0, max_i, i, j] = 1.

    return categorical_one_hot


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


def append_timestamp_to_file(file_name, termination: str = '.pkl') -> str:
    ending = get_ending_with_timestamp(termination=termination)
    ending_len = len(termination)
    return file_name[:-ending_len] + '_' + ending


def get_ending_with_timestamp(termination: str = '.pkl') -> str:
    current_time = str(datetime.datetime.now())
    parts_of_date = current_time.split(sep=' ')
    year_month_day = parts_of_date[0].replace('-', '')
    hour_min = parts_of_date[1].replace(':', '')
    hour_min = hour_min[:4]
    ending = year_month_day + '_' + hour_min + termination
    return ending

import tensorflow as tf


def print_dlgmm_diagnostics(z_kumar, pi, log_a, log_b, mean, log_var,
                            log_px_z, log_pz, log_qpi_x, log_qz_x, loss):
    tf.print("+++++++++++++++++++++++++++++")
    tf.print()
    tf.print()
    tf.print("z_kumar----------------------")
    print_stats(z_kumar)
    tf.print("pi---------------------------")
    print_stats(pi)
    tf.print("log_a------------------------")
    print_stats(log_a)
    tf.print("log_b------------------------")
    print_stats(log_b)
    tf.print("mean-------------------------")
    print_stats(mean)
    tf.print("log_var----------------------")
    print_stats(log_var)
    tf.print("+++++++++++++++++++++++++++++")
    tf.print("Loss")
    tf.print("-----------------------------")
    tf.print(log_px_z)
    tf.print(log_pz)
    tf.print(log_qpi_x)
    tf.print(log_qz_x)
    tf.print(loss)


def print_stats(v):
    tf.print(tf.reduce_max(v))
    tf.print(tf.reduce_mean(v))
    tf.print(tf.reduce_min(v))

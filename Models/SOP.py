import tensorflow as tf
from tensorflow.keras.layers import Flatten, InputLayer, Dense, Reshape
from Utils.Distributions import IGR_I, IGR_SB_Finite, IGR_Planar
from Models.VAENet import PlanarFlowLayer


class SOP(tf.keras.Model):
    def __init__(self, hyper: dict):
        super(SOP, self).__init__()
        self.half_image_w_h = hyper['width_height']
        self.half_image_size = hyper['width_height'][0] * hyper['width_height'][1]
        self.units_per_layer = hyper['units_per_layer']
        self.temp = hyper['temp']
        self.model_type = hyper['model_type']
        self.architecture = hyper['architecture']
        self.var_num = 1
        # self.var_num = 1 if self.model_type == 'GS' else 2
        self.split_sizes_list = [self.units_per_layer for _ in range(self.var_num)]
        self.create_layers_based_on_architecture()

    def create_layers_based_on_architecture(self):
        self.input_layer = InputLayer(input_shape=self.half_image_w_h)
        self.flat_layer = Flatten()
        if self.architecture == 'double_linear':
            self.h1_dense = Dense(units=self.units_per_layer * self.var_num, activation='linear')
            self.h2_dense = Dense(units=self.units_per_layer * self.var_num, activation='linear')
        elif self.architecture == 'triple_linear':
            self.h1_dense = Dense(units=self.units_per_layer * self.var_num, activation='linear')
            self.h2_dense = Dense(units=self.units_per_layer * self.var_num, activation='linear')
            self.h3_dense = Dense(units=self.units_per_layer * self.var_num, activation='linear')
        elif self.architecture == 'nonlinear':
            self.h11_dense = Dense(units=self.units_per_layer * self.var_num, activation='tanh')
            self.h12_dense = Dense(units=self.units_per_layer * self.var_num, activation='linear')
            self.h21_dense = Dense(units=self.units_per_layer * self.var_num, activation='tanh')
            self.h22_dense = Dense(units=self.units_per_layer * self.var_num, activation='linear')
        self.out_dense = Dense(units=self.half_image_size)
        self.reshape_out = Reshape(self.half_image_w_h)
        if self.model_type == 'IGR_Planar':
            self.planar_flow = generate_planar_flow(disc_latent_in=1,
                                                    disc_var_num=self.units_per_layer)
        else:
            self.planar_flow = None

    @tf.function()
    def call(self, x_upper, sample_size=1, use_one_hot=False):
        if self.architecture == 'double_linear':
            logits = self.use_double_linear(x_upper, sample_size, use_one_hot)
        elif self.architecture == 'triple_linear':
            logits = self.use_triple_linear(x_upper, sample_size, use_one_hot)
        else:
            logits = self.use_nonlinear(x_upper, sample_size, use_one_hot)
        return logits

    def use_triple_linear(self, x_upper, sample_size, use_one_hot):
        batch_n, width, height, rgb = x_upper.shape
        logits = tf.TensorArray(dtype=tf.float32, size=sample_size,
                                element_shape=(batch_n, width, height, rgb))
        out = self.h1_dense(self.flat_layer(self.input_layer(x_upper)))
        params_1 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
        for i in range(sample_size):
            z_1 = self.sample_binary(params_1, use_one_hot)
            out = self.h2_dense(z_1)
            params_2 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
            z_2 = self.sample_binary(params_2, use_one_hot)

            out = self.h3_dense(z_2)
            params_3 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
            z_3 = self.sample_binary(params_3, use_one_hot)

            value = self.reshape_out(self.out_dense(z_3))
            logits = logits.write(index=i, value=value)
        logits = tf.transpose(logits.stack(), perm=[1, 2, 3, 4, 0])
        return logits

    def use_double_linear(self, x_upper, sample_size, use_one_hot):
        batch_n, width, height, rgb = x_upper.shape
        logits = tf.TensorArray(dtype=tf.float32, size=sample_size,
                                element_shape=(batch_n, width, height, rgb))
        out = self.h1_dense(self.flat_layer(self.input_layer(x_upper)))
        params_1 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
        for i in range(sample_size):
            z_1 = self.sample_binary(params_1, use_one_hot)
            out = self.h2_dense(z_1)
            params_2 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
            z_2 = self.sample_binary(params_2, use_one_hot)

            value = self.reshape_out(self.out_dense(z_2))
            logits = logits.write(index=i, value=value)
        logits = tf.transpose(logits.stack(), perm=[1, 2, 3, 4, 0])
        return logits

    def use_nonlinear(self, x_upper, sample_size, use_one_hot):
        batch_n, width, height, rgb = x_upper.shape
        logits = tf.TensorArray(dtype=tf.float32, size=sample_size,
                                element_shape=(batch_n, width, height, rgb))
        out = self.h12_dense(self.h11_dense(self.flat_layer(self.input_layer(x_upper))))
        params_1 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
        for i in range(sample_size):
            z_1 = self.sample_binary(params_1, use_one_hot)
            out = self.h22_dense(self.h21_dense(z_1))
            params_2 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
            z_2 = self.sample_binary(params_2, use_one_hot)

            value = self.reshape_out(self.out_dense(z_2))
            logits = logits.write(index=i, value=value)
        logits = tf.transpose(logits.stack(), perm=[1, 2, 3, 4, 0])
        return logits

    def sample_binary(self, params, use_one_hot):
        if self.model_type == 'GS':
            psi = sample_gs_binary(params=params, temp=self.temp)
        elif self.model_type in ['IGR_I', 'IGR_SB', 'IGR_Planar']:
            psi = sample_igr_binary(model_type=self.model_type, params=params, temp=self.temp,
                                    planar_flow=self.planar_flow)
        else:
            raise RuntimeError
        psi = tf.math.round(psi) if use_one_hot else psi
        # making the output be in {-1, 1} as in Maddison et. al 2017
        psi = 2 * psi - 1
        return psi


def sample_gs_binary(params, temp):
    # TODO: add the latex formulas
    log_alpha = params[0]
    unif = tf.random.uniform(shape=log_alpha.shape)
    logistic_sample = tf.math.log(unif) - tf.math.log(1. - unif)
    lam = (log_alpha + logistic_sample) / temp
    psi = tf.math.sigmoid(lam)
    return psi


def sample_igr_binary(model_type, params, temp, planar_flow):
    dist = get_igr_dist(model_type, params, temp, planar_flow)
    dist.generate_sample()
    lam = tf.transpose(dist.lam[:, 0, 0, :], perm=[0, 1])
    psi = tf.math.sigmoid(lam)
    return psi


def get_igr_dist(model_type, params, temp, planar_flow):
    # mu, xi = params
    mu = params[0]
    batch_n, num_of_vars = mu.shape
    xi_broad = tf.zeros(shape=(batch_n, 1, 1, num_of_vars))
    mu_broad = tf.reshape(mu, shape=(batch_n, 1, 1, num_of_vars))
    # xi_broad = tf.reshape(xi, shape=(batch_n, 1, 1, num_of_vars))
    if model_type == 'IGR_I':
        dist = IGR_I(mu=mu_broad, xi=xi_broad, temp=temp)
    elif model_type == 'IGR_Planar':
        dist = IGR_Planar(mu=mu_broad, xi=xi_broad, temp=temp, planar_flow=planar_flow)
    elif model_type == 'IGR_SB':
        dist = IGR_SB_Finite(mu=mu_broad, xi=xi_broad, temp=temp)
    else:
        raise ValueError
    return dist


def generate_planar_flow(disc_latent_in, disc_var_num):
    planar_flow = tf.keras.Sequential([InputLayer(input_shape=(disc_latent_in, 1, disc_var_num)),
                                       PlanarFlowLayer(units=disc_latent_in, var_num=disc_var_num),
                                       PlanarFlowLayer(units=disc_latent_in, var_num=disc_var_num)])
    return planar_flow


def revert_samples_to_last_dim(a, sample_size):
    batch_n, width, height, rgb = a.shape
    new_shape = (int(batch_n / sample_size), width, height, rgb, sample_size)
    a = tf.reshape(a, shape=new_shape)
    return a


def brodcast_samples_to_batch(x_upper, sample_size):
    batch_n, width, height, rgb = x_upper.shape
    x_upper_broad = brodcast_to_sample_size(x_upper, sample_size=sample_size)
    x_upper_broad = tf.reshape(x_upper_broad, shape=(batch_n * sample_size, width, height, rgb))
    return x_upper_broad


def brodcast_to_sample_size(a, sample_size):
    original_shape = a.shape
    newshape = original_shape + (1,)
    broad_shape = original_shape + (sample_size,)

    a = tf.reshape(a, shape=newshape)
    a = tf.broadcast_to(a, shape=broad_shape)
    return a

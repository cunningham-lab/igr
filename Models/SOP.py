import tensorflow as tf
from Utils.Distributions import IGR_I, IGR_SB_Finite, IGR_Planar
from Models.VAENet import PlanarFlowLayer


class SOP(tf.keras.Model):
    def __init__(self, hyper: dict):
        super(SOP, self).__init__()
        self.half_image_w_h = hyper['width_height']
        self.half_image_size = hyper['width_height'][0] * hyper['width_height'][1]
        self.units_per_layer = 200
        self.temp = hyper['temp']
        self.model_type = hyper['model_type']
        self.var_num = 1 if self.model_type == 'GS' else 2
        self.split_sizes_list = [self.units_per_layer for _ in range(self.var_num)]

        self.layer0 = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(units=self.units_per_layer * self.var_num)
        self.layer2 = tf.keras.layers.Dense(units=self.units_per_layer * self.var_num)
        self.layer3 = tf.keras.layers.Dense(units=self.half_image_size)
        self.layer4 = tf.keras.layers.Reshape(self.half_image_w_h)
        if self.model_type == 'IGR_Planar':
            self.planar_flow = generate_planar_flow(disc_latent_in=1, disc_var_num=self.units_per_layer)
        else:
            self.planar_flow = None

    def call(self, x_upper, sample_size=1, discretized=False):
        batch_n, width, height, rgb = x_upper.shape
        x_upper_broad = brodcast_to_sample_size(x_upper, sample_size=sample_size)
        x_upper_broad = tf.reshape(x_upper_broad, shape=(batch_n * sample_size, width, height, rgb))

        out = self.layer1(self.layer0(x_upper))
        params_1 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
        z_1 = self.sample_bernoulli(params_1, discretized=discretized)

        out = self.layer2(z_1)
        params_2 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
        z_2 = self.sample_bernoulli(params_2, discretized=discretized)

        out = self.layer3(z_2)
        logits = self.layer4(out)
        return logits

    def sample_bernoulli(self, params, discretized):
        if self.model_type == 'GS':
            psi = sample_gs_bernoulli(params=params, temp=self.temp, discretized=discretized)
        elif self.model_type in ['IGR_I', 'IGR_SB', 'IGR_Planar']:
            psi = sample_igr_bernoulli(model_type=self.model_type, params=params, temp=self.temp,
                                       discretized=discretized, planar_flow=self.planar_flow)
        else:
            raise RuntimeError
        return psi


def sample_gs_bernoulli(params, temp, discretized):
    log_alpha_broad = params[0]
    unif = tf.random.uniform(shape=log_alpha_broad.shape)
    gumbel = -tf.math.log(-tf.math.log(unif + 1.e-20))
    lam = (log_alpha_broad + gumbel) / temp
    psi = project_to_vertices(lam=lam, discretized=discretized)
    return psi


def sample_igr_bernoulli(model_type, params, temp, discretized, planar_flow):
    dist = get_igr_dist(model_type, params, temp, planar_flow)
    dist.generate_sample()
    psi = project_to_vertices(lam=dist.lam[:, 0, 0, :], discretized=discretized)
    return psi


def get_igr_dist(model_type, params, temp, planar_flow):
    mu, xi = params
    batch_n, num_of_vars = mu.shape
    mu_broad = tf.reshape(mu, shape=(batch_n, 1, 1, num_of_vars))
    xi_broad = tf.reshape(xi, shape=(batch_n, 1, 1, num_of_vars))
    if model_type == 'IGR_I':
        dist = IGR_I(mu=mu_broad, xi=xi_broad, temp=temp)
    elif model_type == 'IGR_Planar':
        dist = IGR_Planar(mu=mu_broad, xi=xi_broad, temp=temp, planar_flow=planar_flow)
    elif model_type == 'IGR_SB':
        dist = IGR_SB_Finite(mu=mu_broad, xi=xi_broad, temp=temp)
    else:
        raise ValueError
    return dist


def project_to_vertices(lam, discretized):
    psi = tf.math.sigmoid(lam)
    if discretized:
        psi = tf.math.round(lam)
    return psi


def brodcast_to_sample_size(a, sample_size):
    original_shape = a.shape
    newshape = original_shape + (1,)
    broad_shape = original_shape + (sample_size,)
    a = tf.reshape(a, shape=newshape)
    a = tf.broadcast_to(a, shape=broad_shape)
    return a


def generate_planar_flow(disc_latent_in, disc_var_num):
    planar_flow = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(disc_latent_in, 1, disc_var_num)),
        PlanarFlowLayer(units=disc_latent_in, var_num=disc_var_num),
        PlanarFlowLayer(units=disc_latent_in, var_num=disc_var_num)])
    return planar_flow

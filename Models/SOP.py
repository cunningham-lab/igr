import tensorflow as tf


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
        elif self.model_type == 'IGR':
            psi = sample_igr_bernoulli(params=params, temp=self.temp, discretized=discretized)
        else:
            raise RuntimeError
        return psi


def sample_gs_bernoulli(params, temp, discretized):
    log_alpha_broad = params[0]
    # log_alpha = params[0]
    # log_alpha_broad = brodcast_to_sample_size(a=log_alpha, sample_size=sample_size)
    unif = tf.random.uniform(shape=log_alpha_broad.shape)
    gumbel = -tf.math.log(-tf.math.log(unif + 1.e-20))
    lam = (log_alpha_broad + gumbel) / temp
    psi = project_to_vertices(lam=lam, discretized=discretized)
    return psi


def sample_igr_bernoulli(params, temp, discretized):
    mu_broad, xi_broad = params
    # mu, xi = params
    # mu_broad = brodcast_to_sample_size(mu, sample_size=sample_size)
    # xi_broad = brodcast_to_sample_size(xi, sample_size=sample_size)
    epsilon = tf.random.normal(shape=mu_broad.shape)
    sigma = tf.math.exp(xi_broad)
    lam = (mu_broad + sigma * epsilon) / temp
    psi = project_to_vertices(lam=lam, discretized=discretized)
    return psi


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

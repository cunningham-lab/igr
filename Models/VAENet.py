import tensorflow as tf
from Utils.general import append_timestamp_to_file
from os import environ as os_env
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VAENet(tf.keras.Model):
    def __init__(self, hyper: dict, image_shape: tuple = (28, 28, 1)):
        super(VAENet, self).__init__()
        self.cont_latent_n = hyper['latent_norm_n']
        self.cont_var_num = hyper['num_of_norm_var']
        self.cont_param_num = hyper['num_of_norm_param']
        self.disc_latent_n = hyper['latent_discrete_n']
        self.disc_var_num = hyper['num_of_discrete_var']
        self.disc_param_num = hyper['num_of_discrete_param']
        self.architecture_type = hyper['architecture']
        self.batch_size = hyper['batch_n']
        self.model_name = hyper['dataset_name']
        self.log_px_z_params_num = 1 if self.model_name == 'mnist' else 2
        # self.log_px_z_params_num = 1 if self.model_name == 'celeb_a' else 1

        self.latent_dim_in = (self.cont_param_num * self.cont_latent_n * self.cont_var_num +
                              self.disc_param_num * self.disc_latent_n * self.disc_var_num)
        self.latent_dim_out = (self.cont_var_num * self.cont_latent_n +
                               self.disc_var_num * self.disc_latent_n)
        self.split_sizes_list = [self.cont_latent_n * self.cont_var_num for _ in range(self.cont_param_num)]
        self.split_sizes_list += [self.disc_latent_n * self.disc_var_num for _ in range(self.disc_param_num)]
        self.num_var = (self.cont_var_num, self.disc_var_num)

        self.image_shape = image_shape
        self.inference_net = tf.keras.Sequential
        self.generative_net = tf.keras.Sequential
        self.sop_net = tf.keras.Sequential
        self.planar_flow = tf.keras.Sequential

    def construct_architecture(self):
        if self.architecture_type == 'dense':
            self.generate_dense_inference_net()
            self.generate_dense_generative_net()
        elif self.architecture_type == 'dense_nf':
            self.generate_dense_inference_net()
            self.generate_planar_flow()
            self.generate_dense_generative_net()
        elif self.architecture_type == 'conv':
            self.generate_convolutional_inference_net()
            self.generate_convolutional_generative_net()
        elif self.architecture_type == 'conv_jointvae':
            if self.model_name == 'celeb_a' or self.model_name == 'fmnist':
                self.generate_convolutional_inference_net_jointvae_celeb_a()
                self.generate_convolutional_generative_net_jointvae_celeb_a()
            else:
                self.generate_convolutional_inference_net_jointvae()
                self.generate_convolutional_generative_net_jointvae()

    # -------------------------------------------------------------------------------------------------------
    def generate_dense_inference_net(self):
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.image_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.image_shape[0] * self.image_shape[1] * self.image_shape[2],
                                  activation='relu'),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=self.latent_dim_in),
        ])

    def generate_dense_generative_net(self):
        if self.model_name == 'celeb_a' or self.model_name == 'fmnist':
            activation_type = 'elu'
        elif self.model_name == 'mnist':
            activation_type = 'linear'
        else:
            raise RuntimeError
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim_out,)),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=self.image_shape[0] * self.image_shape[1] * self.image_shape[2] *
                                  self.log_px_z_params_num, activation=activation_type),
            tf.keras.layers.Reshape(target_shape=(self.image_shape[0], self.image_shape[1],
                                                  self.image_shape[2] * self.log_px_z_params_num))
        ])

    def generate_planar_flow(self):
        self.planar_flow = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.disc_latent_n, 1, self.disc_var_num)),
            PlanarFlowLayer(units=self.disc_latent_n, var_num=self.disc_var_num),
            PlanarFlowLayer(units=self.disc_latent_n, var_num=self.disc_var_num)])

    # -------------------------------------------------------------------------------------------------------
    def generate_convolutional_inference_net_jointvae(self):
        input_layer = tf.keras.layers.Input(shape=self.image_shape)
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(input_layer)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(conv)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(conv)
        if self.image_shape[0] == 64:
            conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                          activation='relu', padding='same')(conv)
        flat_layer = tf.keras.layers.Flatten()(conv)
        dense_layer = tf.keras.layers.Dense(units=256, activation='relu')(flat_layer)
        if len(self.split_sizes_list) == 1:
            concat_layer = tf.keras.layers.Dense(units=self.split_sizes_list[0])(dense_layer)
        else:
            params_layer = [tf.keras.layers.Dense(units=u)(dense_layer) for u in self.split_sizes_list]
            concat_layer = tf.keras.layers.Concatenate()(params_layer)
        self.inference_net = tf.keras.Model(inputs=[input_layer], outputs=[concat_layer])

    def generate_convolutional_generative_net_jointvae(self):
        output_layer = tf.keras.layers.Input(shape=(self.latent_dim_out,))
        layer = tf.keras.layers.Dense(units=256, activation='relu')(output_layer)
        layer = tf.keras.layers.Dense(units=4 * 4 * 64, activation='relu')(layer)
        layer = tf.keras.layers.Reshape(target_shape=(4, 4, 64))(layer)
        if self.image_shape[0] == 64:
            layer = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                    activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=self.image_shape[2], kernel_size=(4, 4),
                                                strides=(2, 2), padding='same')(layer)
        self.generative_net = tf.keras.Model(inputs=[output_layer], outputs=[layer])

    # -------------------------------------------------------------------------------------------------------
    def generate_convolutional_inference_net_jointvae_celeb_a(self):
        input_layer = tf.keras.layers.Input(shape=self.image_shape)
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(input_layer)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(conv)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(conv)
        if self.image_shape[0] == 64:
            conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                          activation='relu', padding='same')(conv)
        flat_layer = tf.keras.layers.Flatten()(conv)
        dense_layer = tf.keras.layers.Dense(units=256, activation='relu')(flat_layer)
        if len(self.split_sizes_list) == 1:
            concat_layer = tf.keras.layers.Dense(units=self.split_sizes_list[0])(dense_layer)
        else:
            params_layer = [tf.keras.layers.Dense(units=u)(dense_layer) for u in self.split_sizes_list]
            concat_layer = tf.keras.layers.Concatenate()(params_layer)
        self.inference_net = tf.keras.Model(inputs=[input_layer], outputs=[concat_layer])

    def generate_convolutional_generative_net_jointvae_celeb_a(self):
        output_layer = tf.keras.layers.Input(shape=(self.latent_dim_out,))
        layer = tf.keras.layers.Dense(units=256, activation='relu')(output_layer)
        layer = tf.keras.layers.Dense(units=4 * 4 * 64, activation='relu')(layer)
        layer = tf.keras.layers.Reshape(target_shape=(4, 4, 64))(layer)
        if self.image_shape[0] == 64:
            layer = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                    activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=self.image_shape[2] * self.log_px_z_params_num,
                                                kernel_size=(4, 4), strides=(2, 2), padding='same',
                                                activation='elu')(layer)
        self.generative_net = tf.keras.Model(inputs=[output_layer], outputs=[layer])

    # -------------------------------------------------------------------------------------------------------
    def generate_convolutional_inference_net(self):
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.image_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
                                   activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
                                   activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim_in),
        ])

    def generate_convolutional_generative_net(self):
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_out_n,)),
            tf.keras.layers.Dense(units=7 * 7 * 32, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ])

    # -------------------------------------------------------------------------------------------------------
    # Encoding and Decoding Methods
    def encode(self, x):
        params = self.split_network_parameters(x=x)
        return params

    def decode(self, z):
        logits = tf.split(self.generative_net(z), num_or_size_splits=self.log_px_z_params_num, axis=3)
        return logits

    def split_network_parameters(self, x):
        params = tf.split(self.inference_net(x), num_or_size_splits=self.split_sizes_list, axis=1)
        reshaped_params = []
        for idx, param in enumerate(params):
            batch_size = param.shape[0]
            if self.disc_var_num > 1:
                param = tf.reshape(param,
                                   shape=(batch_size, self.disc_latent_n, 1, self.disc_var_num))
            else:
                param = tf.reshape(param,
                                   shape=(batch_size, self.split_sizes_list[idx], 1, self.disc_var_num))
            reshaped_params.append(param)
        return reshaped_params
    # -------------------------------------------------------------------------------------------------------


class PlanarFlowLayer(tf.keras.layers.Layer):

    def __init__(self, units, var_num, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.var_num = var_num

    def build(self, input_shape):
        self.w = self.add_weight(shape=(1, self.units, 1, self.var_num), initializer='random_normal',
                                 trainable=True, name='w')
        self.b = self.add_weight(shape=(1, 1, 1, self.var_num), initializer='zeros', trainable=True, name='b')
        self.u = self.add_weight(shape=(1, self.units, 1, self.var_num), initializer='random_normal',
                                 trainable=True, name='u')
        super().build(input_shape)

    def call(self, inputs):
        prod_wTu = tf.math.reduce_sum(self.w * self.u, axis=1)
        alpha = -1 + tf.math.softplus(prod_wTu) - prod_wTu
        u_tilde = self.u + alpha * self.w / tf.linalg.norm(self.w)

        batch_n, n_required, sample_size, var_num = inputs.shape
        if batch_n is None:
            batch_n = 1

        w_broad = tf.broadcast_to(self.w, shape=(batch_n, n_required, sample_size, var_num))
        u_tilde_broad = tf.broadcast_to(u_tilde, shape=(batch_n, n_required, sample_size, var_num))

        prod_wTinputs = tf.math.reduce_sum(inputs * w_broad, axis=1, keepdims=True)
        tanh = tf.math.tanh(prod_wTinputs + self.b)
        output = inputs + u_tilde_broad * tanh
        return output

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'units': self.units, 'var_num': self.var_num}


def determine_path_to_save_results(model_type, dataset_name):
    results_path = './Log/' + dataset_name + '_' + model_type + append_timestamp_to_file('', termination='')
    return results_path


def setup_model(hyper, image_size):
    model = VAENet(hyper=hyper, image_shape=image_size)
    model.construct_architecture()
    return model

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def load_vae_dataset(dataset_name, batch_n, epochs, run_with_sample=True, architecture='dense'):
    images_to_display = [10, 25, 5, 29, 1, 35, 18, 30, 6, 19, 15, 23, 11, 21, 17, 26, 344, 3567, 9, 20]
    if dataset_name == 'fmnist':
        if architecture == 'conv_jointvae':
            _, np_test_images = fetch_and_binarize_mnist_data(use_fashion=True)
            image_size = (32, 32, 1)
            data = load_mnist_data(batch_n=batch_n, epochs=epochs, run_with_sample=run_with_sample,
                                   resize=True, use_fashion=True)
            np_test_images = tf.image.resize(np_test_images, size=image_size[0:2]).numpy()
        else:
            _, np_test_images = fetch_and_binarize_mnist_data(use_fashion=True)
            image_size = (28, 28, 1)
            data = load_mnist_data(batch_n=batch_n, epochs=epochs, run_with_sample=run_with_sample,
                                   resize=False, use_fashion=True)
        np_test_images = np_test_images[images_to_display, :, :, :]
        train_dataset, test_dataset, batch_n, epochs = data
    elif dataset_name == 'mnist':
        if architecture == 'conv_jointvae':
            _, np_test_images = fetch_and_binarize_mnist_data()
            image_size = (32, 32, 1)
            data = load_mnist_data(batch_n=batch_n, epochs=epochs, run_with_sample=run_with_sample,
                                   resize=True)
            np_test_images = tf.image.resize(np_test_images, size=image_size[0:2]).numpy()
        else:
            _, np_test_images = fetch_and_binarize_mnist_data()
            image_size = (28, 28, 1)
            data = load_mnist_data(batch_n=batch_n, epochs=epochs, run_with_sample=run_with_sample,
                                   resize=False)
        np_test_images = np_test_images[images_to_display, :, :, :]
        train_dataset, test_dataset, batch_n, epochs = data
    elif dataset_name == 'celeb_a':
        image_size = (64, 64, 3)
        # image_size = (218, 178, 3)
        pd = ProcessData(dataset_name=dataset_name, run_with_sample=run_with_sample, image_size=image_size)
        output = pd.generate_train_and_test_partitions(batch_size=batch_n, epochs=epochs,
                                                       test_size=19962)
        split_data, batch_size, epochs, np_test_images = output
        train_dataset, test_dataset = split_data
    elif dataset_name == 'omniglot':
        image_size = (96, 96, 1)
        pd = ProcessData(dataset_name=dataset_name, run_with_sample=run_with_sample, image_size=image_size)
        output = pd.generate_train_and_test_partitions(batch_size=batch_n, epochs=epochs,
                                                       test_size=13180)
        split_data, batch_size, epochs, np_test_images = output
        train_dataset, test_dataset = split_data
    else:
        raise RuntimeError
    iter_per_epoch = determine_iter_per_epoch(dataset_name=dataset_name, run_with_sample=run_with_sample,
                                              batch_n=batch_n)
    return train_dataset, test_dataset, np_test_images, batch_n, epochs, image_size, iter_per_epoch


def load_mnist_data(batch_n, epochs, use_fashion=False, run_with_sample=False, resize=False):
    train_images, test_images = fetch_and_binarize_mnist_data(use_fashion=use_fashion)
    if run_with_sample:
        train_buffer, test_buffer, batch_n, epochs = 60, 10, 5, 10
        train_images, test_images = train_images[:60, :, :, :], test_images[:10, :, :, :]
    else:
        train_buffer, test_buffer, batch_n, epochs = 60_000, 10_000, batch_n, epochs

    if resize:
        train_images = tf.image.resize(train_images, size=(32, 32))
        test_images = tf.image.resize(test_images, size=(32, 32))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_buffer).batch(batch_n)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_buffer).batch(batch_n)
    return train_dataset, test_dataset, batch_n, epochs


def load_mnist_sop_data(batch_n, epochs, run_with_sample=False):
    train_images, test_images = fetch_and_binarize_mnist_data()
    if run_with_sample:
        train_buffer, test_buffer = 60, 10
        train_images = train_images[:train_buffer, :, :, :]
        test_images = test_images[:test_buffer, :, :, :]
    else:
        train_buffer, test_buffer = 60_000, 10_000

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_buffer).batch(batch_n)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_buffer).batch(batch_n)
    return train_dataset, test_dataset


def fetch_and_binarize_mnist_data(use_fashion=False, output_labels=False):
    if use_fashion:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images, test_images = reshape_binarize_and_scale_images((train_images, test_images),
                                                                      round_images=False)
    else:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images, test_images = reshape_binarize_and_scale_images((train_images, test_images))

    if output_labels:
        return (train_images, train_labels), (test_images, test_labels)
    else:
        return train_images, test_images


class ProcessData:

    def __init__(self, dataset_name, image_size: tuple = (28, 28, 1), run_with_sample=False):
        self.dataset_name = dataset_name
        self.run_with_sample = run_with_sample
        self.image_size = image_size

    def generate_train_and_test_partitions(self, batch_size, epochs, test_size):
        data = fetch_data_via_tf_datasets(dataset_name=self.dataset_name)
        buffer_size, batch_size, epochs = determine_buffer_re_assign_batch_and_epochs(
            self.run_with_sample, batch_size=batch_size, epochs=epochs)
        split_names, split_data, test_position = ['train', 'test'], [], 1
        for split in split_names:
            processed_data = self.preprocess(data_split=data[split], buffer_size=buffer_size,
                                             batch_size=batch_size)
            split_data.append(processed_data)
        np_test_images = self.fetch_test_numpy_images(test_ds=split_data[test_position], test_size=test_size)
        return split_data, batch_size, epochs, np_test_images

    def preprocess(self, data_split, buffer_size, batch_size):
        if self.dataset_name == 'omniglot':
            return data_split.map(preprocess_omniglot).shuffle(buffer_size).batch(batch_size)
        elif self.dataset_name == 'celeb_a':
            return data_split.map(preprocess_celeb_a).shuffle(buffer_size).batch(batch_size)
        else:
            raise RuntimeError

    def fetch_test_numpy_images(self, test_ds, test_size):
        test_images = iterate_over_dataset_container(data_iterable=test_ds, test_size=test_size,
                                                     image_size=self.image_size)
        images_to_display = [10, 25, 5, 29, 1, 35, 18, 30, 6, 19, 15, 23, 11, 21, 17, 26, 344, 3567, 9, 20]
        return test_images[images_to_display, :, :, :]


def fetch_data_via_tf_datasets(dataset_name):
    builder = tfds.builder(name=dataset_name)
    builder.download_and_prepare()
    data = builder.as_dataset(shuffle_files=False)
    return data


def fetch_numpy_data_via_tf_datasets(dataset_name):
    np_data = tfds.as_numpy(dataset=fetch_data_via_tf_datasets(dataset_name=dataset_name))
    return np_data


def determine_buffer_re_assign_batch_and_epochs(run_with_sample, batch_size, epochs):
    if run_with_sample:
        buffer, batch_size, epochs = 60, 5, 10
    else:
        buffer = 1000
    return buffer, batch_size, epochs


def preprocess_celeb_a(example):
    example['image'] = binarize_tensor(example['image'])
    example['image'] = tf.image.resize(images=example['image'], size=(64, 64))
    return example['image']


def preprocess_omniglot(example):
    example['image'] = binarize_tensor(example['image'])
    example['image'] = round_tensor(example['image'])
    example['image'] = crop_tensor(example['image'], limit=96)
    return example['image']


def binarize_tensor(tensor):
    new_tensor = tf.cast(tensor, dtype=tf.float32)
    new_tensor = new_tensor / 255.
    return new_tensor


def round_tensor(tensor):
    rounded_tensor = tf.cast(tensor, dtype=tf.int32)
    rounded_tensor = tf.cast(rounded_tensor, dtype=tf.float32)
    return rounded_tensor


def crop_tensor(tensor, limit: int):
    tensor = tf.reshape(tensor[:limit, :limit, 0], shape=(limit, limit, 1))
    return tensor


def iterate_over_dataset_container(data_iterable, test_size, image_size):
    images = np.zeros(shape=(test_size,) + image_size)
    i = 0
    for image in data_iterable:
        for batch_idx in range(image.shape[0]):
            images[i, :, :, :] = image[batch_idx, :, :, :]
            i += 1
    return images


def determine_iter_per_epoch(dataset_name, run_with_sample, batch_n):
    if dataset_name == 'fmnist' or dataset_name == 'mnist':
        num_of_data_points = 60_000
        iter_per_epoch = num_of_data_points // batch_n
    elif dataset_name == 'omniglot':
        if run_with_sample:
            iter_per_epoch = 5
        else:
            num_of_data_points = 19_280
            iter_per_epoch = num_of_data_points // batch_n
    elif dataset_name == 'celeb_a':
        if run_with_sample:
            iter_per_epoch = 5
        else:
            num_of_data_points = 19_962
            iter_per_epoch = num_of_data_points // batch_n
    else:
        raise RuntimeError
    return iter_per_epoch


def get_dataset_from_tf(name: str):
    builder = tfds.builder(name=name)
    builder.download_and_prepare()
    ds = builder.as_dataset(shuffle_files=False)
    ds_numpy = tfds.as_numpy(dataset=ds)
    return ds_numpy


def put_into_numpy_arrays(dataset):
    test_images = np.zeros(shape=(13180, 96, 96))
    train_images = np.zeros(shape=(19280, 96, 96))
    for i, test_image in enumerate(dataset['test']):
        test_images[i, :, :] = test_image['image'][:96, :96, 0]

    for i, train_image in enumerate(dataset['train']):
        train_images[i, :, :] = train_image['image'][:96, :96, 0]
    return train_images, test_images


def reshape_binarize_and_scale_images(images, round_images=True):
    output = []
    for im in images:
        im = np.reshape(im, newshape=im.shape + (1,)).astype('float32')
        im /= 255.
        if round_images:
            im[im >= 0.5] = 1.
            im[im < 0.5] = 0.
        output.append(im)
    return output

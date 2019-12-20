import tensorflow as tf


class Classifier(tf.keras.Model):

    def __init__(self, shape_of_input, shape_of_output):
        super(Classifier, self).__init__()
        self.classifier_net = tf.keras.Sequential
        self.shape_of_input = shape_of_input
        self.shape_of_output = shape_of_output

    def make_predictions(self, x):
        pyx = self.classifier_net(x)
        return pyx


class DenseClassifier(Classifier):

    def __init__(self, shape_of_input, shape_of_output):
        super().__init__(shape_of_input=shape_of_input, shape_of_output=shape_of_output)

    def construct_classifier_net(self):
        self.classifier_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.shape_of_input),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.shape_of_output, activation='softmax'),
        ])


class ConvolutionalClassifier(Classifier):

    def __init__(self, shape_of_input, shape_of_output):
        super(Classifier, self).__init__(shape_of_input=shape_of_input, shape_of_output=shape_of_output)

    def construct_classifier_net(self):
        self.classifier_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.shape_of_input),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=(3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.shape_of_output, activation='softmax'),
        ])
